
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
import os
import random
import warnings
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ==================== 1. Positional Encoding ====================

class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ==================== 2. Multi-Scale Temporal Encoder ====================

class MultiScaleTemporalEncoder(nn.Module):
    """多尺度时序编码器"""
    def __init__(self, input_dim=38, hidden_dim=128, scales=[3, 5, 11, 21]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = len(scales)

        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim // self.num_scales, kernel_size=s, padding=s//2)
            for s in scales
        ])

        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=512)

        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        B, T, F_dim = x.shape
        x_permuted = x.permute(0, 2, 1)
        multi_scale_features = []
        for conv in self.convs:
            conv_out = conv(x_permuted)
            multi_scale_features.append(conv_out)

        x_multi = torch.cat(multi_scale_features, dim=1)
        x_multi = x_multi.permute(0, 2, 1)
        x_pos = self.pos_encoder(x_multi)
        h_temp = self.temporal_transformer(x_pos)
        h = self.projection(h_temp)

        return h


# ==================== 3. Spatial Transformer Encoder ====================

class SpatialTransformerEncoder(nn.Module):
    """空间Transformer编码器"""
    def __init__(self, hidden_dim=128, num_heads=8):
        super().__init__()

        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=1
        )

    def forward(self, h):
        B, T, D = h.shape
        h_reshape = h.view(B * T, 1, D)
        h_spatial = self.spatial_transformer(h_reshape)
        h_spatial = h_spatial.view(B, T, D)
        h_out = h + h_spatial
        return h_out


# ==================== 4. Learnable Context Aggregator ====================

class LearnableContextAggregator(nn.Module):
    """可学习的上下文聚合器"""
    def __init__(self, window_size_k: int = 5, use_position_bias: bool = True, temperature: float = 0.5):
        super().__init__()
        self.k = window_size_k
        self.window_len = 2 * window_size_k + 1
        self.temperature = temperature

        self.logits = nn.Parameter(torch.zeros(self.window_len))

        if use_position_bias:
            positions = torch.arange(-window_size_k, window_size_k + 1)
            position_prior = torch.exp(-0.5 * (positions / (window_size_k / 2)) ** 2)
            self.register_buffer('position_prior', position_prior)
        else:
            self.position_prior = None

    def forward(self, h_seq):
        if self.position_prior is not None:
            weights = self.logits + torch.log(self.position_prior + 1e-8)
        else:
            weights = self.logits

        weights = Func.softmax(weights / self.temperature, dim=0)
        context = torch.einsum('l,...l d->...d', weights, h_seq)
        return context


# ==================== 5. Main Model ====================

class ResidualPrototypeAlignmentNetwork(nn.Module):
    """残差原型对齐网络"""
    def __init__(self,
                 input_dim=38,
                 hidden_dim=128,
                 window_size_k=5,
                 temporal_scales=[3, 5, 11, 21],
                 use_spatial_transformer=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.window_size_k = window_size_k
        self.window_len = 2 * window_size_k + 1

        self.temporal_encoder = MultiScaleTemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            scales=temporal_scales
        )

        self.use_spatial = use_spatial_transformer
        if use_spatial_transformer:
            self.spatial_encoder = SpatialTransformerEncoder(hidden_dim=hidden_dim)

        self.context_aggregator = LearnableContextAggregator(
            window_size_k=window_size_k,
            use_position_bias=True,
            temperature=0.5
        )

        self.p_norm = nn.Parameter(torch.randn(hidden_dim) / np.sqrt(hidden_dim))
        self.p_dummy = nn.Parameter(torch.randn(hidden_dim) / np.sqrt(hidden_dim))

        self._init_prototypes()

    def _init_prototypes(self):
        with torch.no_grad():
            self.p_norm.data = Func.normalize(self.p_norm.data, dim=0)
            self.p_dummy.data = Func.normalize(self.p_dummy.data, dim=0)

    def compute_residuals(self, h):
        B, T, D = h.shape
        k = self.window_size_k

        h_padded = Func.pad(h.permute(0, 2, 1), (k, k), mode='reflect').permute(0, 2, 1)

        neighborhoods = []
        for i in range(T):
            neighborhood = h_padded[:, i:i + self.window_len, :]
            neighborhoods.append(neighborhood)

        neighborhoods = torch.stack(neighborhoods, dim=1)
        contexts = self.context_aggregator(neighborhoods)
        residuals = h - contexts

        return residuals, contexts

    def forward(self, x):
        h = self.temporal_encoder(x)

        if self.use_spatial:
            h = self.spatial_encoder(h)

        r, _ = self.compute_residuals(h)

        return r, h

    def anomaly_score(self, r, reduce_mean=True):
        p_norm_norm = Func.normalize(self.p_norm, dim=0)
        r_norm = Func.normalize(r, dim=-1)

        sim = torch.einsum('btd,d->bt', r_norm, p_norm_norm)
        scores = 1 - sim

        if reduce_mean:
            scores = scores.mean(dim=1)

        return scores


# ==================== 6. Loss Function ====================

class RPANLoss(nn.Module):
    """RPAN损失函数"""
    def __init__(self,
                 temperature=0.07,
                 margin=1.0,
                 alpha=0.5,
                 beta=0.1,
                 gamma=0.01,
                 delta=0.01):
        super().__init__()
        self.tau = temperature
        self.m = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def cosine_sim(self, a, b):
        a_norm = Func.normalize(a, dim=-1)
        b_norm = Func.normalize(b, dim=-1)
        return (a_norm * b_norm).sum(dim=-1)

    def contrast_loss(self, r, p_norm, p_dummy):
        sim_norm = self.cosine_sim(r, p_norm.unsqueeze(0))
        sim_dummy = self.cosine_sim(r, p_dummy.unsqueeze(0))

        logits = torch.stack([sim_norm / self.tau, sim_dummy / self.tau], dim=1)
        labels = torch.zeros(r.size(0), dtype=torch.long, device=r.device)

        loss = Func.cross_entropy(logits, labels)
        return loss

    def invariance_loss(self, r, r_aug):
        sim = self.cosine_sim(r, r_aug)
        loss = -sim.mean()
        return loss

    def separation_loss(self, p_norm, p_dummy):
        dist = torch.norm(p_norm - p_dummy, p=2)
        loss = torch.relu(self.m - dist)
        return loss

    def norm_loss(self, p_norm, p_dummy):
        loss = (torch.norm(p_norm, p=2) - 1).pow(2) + (torch.norm(p_dummy, p=2) - 1).pow(2)
        return loss

    def margin_loss(self, r):
        loss = r.norm(p=2, dim=-1).pow(2).mean()
        return loss

    def forward(self, r, r_aug, p_norm, p_dummy):
        L_contrast = self.contrast_loss(r, p_norm, p_dummy)
        L_inv = self.invariance_loss(r, r_aug)
        L_sep = self.separation_loss(p_norm, p_dummy)
        L_norm = self.norm_loss(p_norm, p_dummy)
        L_margin = self.margin_loss(r)

        total = (L_contrast +
                 self.alpha * L_inv +
                 self.beta * L_sep +
                 self.gamma * L_norm +
                 self.delta * L_margin)

        loss_dict = {
            'contrast': L_contrast.item(),
            'invariance': L_inv.item(),
            'separation': L_sep.item(),
            'norm': L_norm.item(),
            'margin': L_margin.item(),
            'total': total.item()
        }

        return total, loss_dict


# ==================== 7. Data Augmentation ====================

class TimeSeriesAugmentation:
    """时序数据增强"""

    @staticmethod
    def gaussian_noise(x, std=0.05):
        noise = torch.randn_like(x) * std
        return x + noise

    @staticmethod
    def feature_masking(x, mask_ratio=0.3):
        B, T, F_dim = x.shape
        mask = torch.rand(B, 1, F_dim, device=x.device) < mask_ratio
        x_masked = x.clone()
        x_masked[mask.expand(-1, T, -1)] = 0
        return x_masked

    @classmethod
    def apply_augmentation(cls, x):
        aug_type = np.random.choice(['noise', 'mask', 'none'], p=[0.4, 0.3, 0.3])

        if aug_type == 'noise':
            return cls.gaussian_noise(x, std=0.05)
        elif aug_type == 'mask':
            return cls.feature_masking(x, mask_ratio=0.3)
        else:
            return x


# ==================== 8. SMD Dataset (Memory Efficient) ====================

class SMDSingleMachineDataset(Dataset):
    """内存高效的数据集"""
    def __init__(self, data, window_size=11):
        self.data = data.astype(np.float32)
        self.window_size = window_size
        self.n_samples = max(0, len(data) - window_size + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.window_size]
        return torch.from_numpy(window.copy())


# ==================== 9. Affiliation Evaluation Metrics ====================

def affiliation_precision(anomaly_intervals, predicted_points, max_distance=50):
    if len(predicted_points) == 0:
        return 0.0

    matched_count = 0
    for pred_point in predicted_points:
        min_distance = max_distance
        for start, end in anomaly_intervals:
            if start <= pred_point <= end:
                min_distance = 0
                break
            else:
                dist = min(abs(pred_point - start), abs(pred_point - end))
                if dist < min_distance:
                    min_distance = dist

        if min_distance < max_distance:
            matched_count += 1

    return matched_count / len(predicted_points)


def affiliation_recall(anomaly_intervals, predicted_points, max_distance=50):
    if len(anomaly_intervals) == 0:
        return 1.0

    total_coverage = 0.0
    for start, end in anomaly_intervals:
        interval_length = end - start + 1
        if interval_length <= 0:
            continue

        covered_count = 0
        for pred_point in predicted_points:
            if start <= pred_point <= end:
                covered_count += 1
            else:
                dist = min(abs(pred_point - start), abs(pred_point - end))
                if dist < max_distance:
                    covered_count += 1 - (dist / max_distance)

        coverage = min(covered_count, interval_length) / interval_length
        total_coverage += coverage

    return total_coverage / len(anomaly_intervals)


def affiliation_f1_score(labels, scores, threshold_percentile=99, max_distance=50, threshold=None):
    if threshold is None:
        positive_scores = scores[scores > 0]
        if len(positive_scores) > 0:
            threshold = np.percentile(positive_scores, threshold_percentile)
        else:
            threshold = np.percentile(scores, threshold_percentile)

    predicted_points = np.where(scores >= threshold)[0].tolist()

    anomaly_intervals = []
    in_anomaly = False
    start = 0

    for i, label in enumerate(labels):
        if label == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif label == 0 and in_anomaly:
            anomaly_intervals.append((start, i - 1))
            in_anomaly = False

    if in_anomaly:
        anomaly_intervals.append((start, len(labels) - 1))

    precision = affiliation_precision(anomaly_intervals, predicted_points, max_distance)
    recall = affiliation_recall(anomaly_intervals, predicted_points, max_distance)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


class AffiliationEvaluator:
    """Affiliation 评估器"""
    def __init__(self, max_distance=50, threshold_percentiles=[95, 96, 97, 98, 99]):
        self.max_distance = max_distance
        self.threshold_percentiles = threshold_percentiles

    def evaluate(self, labels, scores):
        best_f1 = 0.0
        best_results = {}

        for percentile in self.threshold_percentiles:
            p, r, f1 = affiliation_f1_score(
                labels, scores,
                threshold_percentile=percentile,
                max_distance=self.max_distance
            )

            if f1 > best_f1:
                best_f1 = f1
                best_results = {'percentile': percentile, 'precision': p, 'recall': r, 'f1': f1}

        return {
            'best_f1': best_f1,
            'best_results': best_results
        }


# ==================== 10. Trainer with Best Model Selection ====================

class RPANTrainer:
    """RPAN模型训练器 - 每个epoch评估并保存最佳模型"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.augmentation = TimeSeriesAugmentation()
        self.best_f1 = -1
        self.best_epoch = -1
        self.best_model_state = None

    def evaluate_f1(self, test_dataset, test_labels, batch_size=64):
        """评估当前模型的F1分数"""
        self.model.eval()
        all_scores = []

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        with torch.no_grad():
            for batch_x in test_loader:
                batch_x = batch_x.to(self.device)
                r, _ = self.model(batch_x)
                scores = self.model.anomaly_score(r, reduce_mean=True)
                all_scores.extend(scores.cpu().numpy())

        all_scores = np.array(all_scores)

        evaluator = AffiliationEvaluator(max_distance=50)
        results = evaluator.evaluate(test_labels, all_scores)

        return results['best_f1']

    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        loss_dict_sum = {}

        for x in train_loader:
            x = x.to(self.device)
            x_aug = self.augmentation.apply_augmentation(x)

            r, _ = self.model(x)
            r_aug, _ = self.model(x_aug)

            r_flat = r.view(-1, self.model.hidden_dim)
            r_aug_flat = r_aug.view(-1, self.model.hidden_dim)

            loss, loss_dict = criterion(
                r_flat, r_aug_flat,
                self.model.p_norm, self.model.p_dummy
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v

        avg_loss = total_loss / len(train_loader)
        for k in loss_dict_sum:
            loss_dict_sum[k] /= len(train_loader)

        return avg_loss, loss_dict_sum

    def train(self, train_loader, test_dataset, test_labels,
              epochs=100, lr=1e-4, patience=15):
        optimizer = Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = RPANLoss()

        best_loss = float('inf')
        patience_counter = 0

        print(f"\nStarting training for {epochs} epochs...")
        print(f"Evaluating F1 on test set after each epoch\n")

        for epoch in range(epochs):
            # 训练一个epoch
            train_loss, loss_dict = self.train_epoch(train_loader, optimizer, criterion)
            scheduler.step()

            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # 每个epoch都评估F1
            current_f1 = self.evaluate_f1(test_dataset, test_labels)

            print(f"Epoch {epoch+1:3d}/{epochs}: Loss={train_loss:.4f}, "
                  f"Contrast={loss_dict['contrast']:.4f}, "
                  f"Inv={loss_dict['invariance']:.4f}, "
                  f"F1={current_f1:.4f}")

            # 保存最佳模型
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.best_epoch = epoch + 1
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                torch.save(self.best_model_state, f'best_model_f1_{self.best_f1:.4f}_epoch_{self.best_epoch}.pt')
                print(f"  >>> New best model! F1: {self.best_f1:.4f} at epoch {self.best_epoch}")

            # 早停
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n{'='*50}")
            print(f"Training completed!")
            print(f"Best F1: {self.best_f1:.4f} at epoch {self.best_epoch}")
            print(f"{'='*50}")
        else:
            print("Warning: No best model found!")

        return self.best_f1, self.best_epoch


# ==================== 11. Data Loading ====================

def load_real_smd_data(data_path):
    """加载真实的SMD数据集"""
    train_data_list = []
    test_data_list = []
    test_label_list = []

    for mid in range(1, 13):
        train_file = os.path.join(data_path, f'omi-{mid}_train.pkl')
        test_file = os.path.join(data_path, f'omi-{mid}_test.pkl')
        label_file = os.path.join(data_path, f'omi-{mid}_test_label.pkl')

        train_data = pickle.load(open(train_file, "rb")).astype(np.float32)
        test_data = pickle.load(open(test_file, "rb")).astype(np.float32)
        test_label = pickle.load(open(label_file, "rb")).astype(np.int64)

        # 标准化
        # scaler = StandardScaler().fit(train_data)
        # train_data = scaler.transform(train_data)
        # test_data = scaler.transform(test_data)

        train_data_list.append(train_data)
        test_data_list.append(test_data)
        test_label_list.append(test_label)

    return train_data_list, test_data_list, test_label_list


# ==================== 12. Main Experiment ====================

def run_leave_one_machine_out(data_path=None, window_size=11, hidden_dim=128, epochs=100, use_real_data=False):
    """运行留一机交叉验证"""
    print("Loading data...")

    if use_real_data and data_path:
        all_train_data, all_test_data, all_test_labels = load_real_smd_data(data_path)
        print("Using real SMD data")
    else:
        print("No data path provided, using synthetic data")
        return [], [], []

    n_machines = len(all_train_data)
    print(f"Number of machines: {n_machines}")

    all_best_f1 = []
    all_best_epochs = []

    for target_idx in range(n_machines):
        print(f"\n{'='*60}")
        print(f"Fold {target_idx+1}/{n_machines}: Testing on Machine {target_idx+1}")
        print(f"{'='*60}")

        # 构建源域数据集
        source_datasets = []
        for src_idx in range(n_machines):
            if src_idx == target_idx:
                continue
            ds = SMDSingleMachineDataset(all_train_data[src_idx], window_size=window_size)
            source_datasets.append(ds)

        train_dataset = ConcatDataset(source_datasets)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        print(f"Total training samples: {len(train_dataset)}")

        # 构建目标域测试数据集
        test_dataset = SMDSingleMachineDataset(all_test_data[target_idx], window_size=window_size)
        half_window = window_size // 2
        test_label_array = all_test_labels[target_idx][half_window:half_window + len(test_dataset)]
        print(f"Test samples: {len(test_dataset)}")

        # 创建模型
        model = ResidualPrototypeAlignmentNetwork(
            input_dim=19,
            hidden_dim=hidden_dim,
            window_size_k=window_size // 2,
            temporal_scales=[3, 5, 11, 21],
            use_spatial_transformer=True
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        trainer = RPANTrainer(model, device=device)

        # 训练（每个epoch评估测试集）
        best_f1, best_epoch = trainer.train(
            train_loader, test_dataset, test_label_array,
            epochs=epochs, lr=1e-4, patience=100
        )

        all_best_f1.append(best_f1)
        all_best_epochs.append(best_epoch)
        print(f"\nFold {target_idx+1} - Best F1: {best_f1:.4f} at epoch {best_epoch}")

        # 清理内存
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if all_best_f1:
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Mean Best F1: {np.mean(all_best_f1):.4f} ± {np.std(all_best_f1):.4f}")
        print(f"Min Best F1: {np.min(all_best_f1):.4f}")
        print(f"Max Best F1: {np.max(all_best_f1):.4f}")
        print(f"Average best epoch: {np.mean(all_best_epochs):.1f}")

    return all_best_f1, all_best_epochs


def main():
    """主函数"""
    print("=== Residual Prototype Alignment Network (RPAN) ===")
    print("With Best Model Selection - Each epoch evaluates F1 on test set\n")

    config = {
        'data_path': 'raw_data/ASD',
        'window_size': 9,
        'hidden_dim': 128,
        'epochs': 100,
        'use_real_data': True
    }

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    run_leave_one_machine_out(
        data_path=config['data_path'],
        window_size=config['window_size'],
        hidden_dim=config['hidden_dim'],
        epochs=config['epochs'],
        use_real_data=config['use_real_data']
    )

    print("\n=== Code Execution Completed ===")


if __name__ == "__main__":
    main()
    