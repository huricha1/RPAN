import torch
import torch.nn as nn

from model.sr3_modules.diffusion import MoEConditionalDiffusion


# import MoEConditionalDiffusion from diffusion
class AnomalyDetector(nn.Module):
    """不使用加速采样的完整扩散过程异常检测器"""

    def __init__(self, input_dim, num_experts=5):
        super().__init__()
        self.diffusion = MoEConditionalDiffusion(input_dim, num_experts)

        # 异常评分网络
        self.scorer = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def reconstruct(self, x):
        """完整扩散去噪过程"""
        self.eval()
        with torch.no_grad():
            # 获取MoE权重
            _, expert_weights, attn_weights = self.diffusion.moe(x)
            moe_weights = expert_weights.mean(dim=1)

            # 从纯噪声开始
            noisy_x = torch.randn_like(x[:, -1, :])

            # 完整反向扩散过程
            for t in reversed(range(self.diffusion.diffusion_steps)):
                # 当前时间步
                t_tensor = torch.full((x.size(0),), t, device=x.device)

                # 去噪步骤
                pred_noise = self.diffusion.denoise_step(
                    noisy_x,
                    t_tensor,
                    moe_weights
                )

                # 计算相关参数
                alpha_t = self.diffusion.alphas[t]
                alpha_bar_t = self.diffusion.alpha_bars[t]
                beta_t = self.diffusion.betas[t]
                sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

                # 更新噪声数据
                if t > 0:
                    noise = torch.randn_like(noisy_x)
                else:
                    noise = torch.zeros_like(noisy_x)

                noisy_x = (
                                  sqrt_recip_alpha_t *
                                  (noisy_x - beta_t * pred_noise / sqrt_one_minus_alpha_bar_t)
                          ) + torch.sqrt(beta_t) * noise

            reconstruction = noisy_x
            return reconstruction, moe_weights, attn_weights

    def anomaly_score(self, x):
        """计算异常分数(使用完整去噪过程)"""
        # 重建最后一个时间步
        reconstruction, moe_weights, attn_weights = self.reconstruct(x)
        last_features = x[:, -1, :]

        # 计算特征级重建误差
        feature_errors = torch.abs(last_features - reconstruction)

        # 加权异常分数
        weighted_errors = feature_errors * moe_weights.unsqueeze(-1)

        # 最终异常分数
        anomaly_scores = self.scorer(
            torch.cat([
                weighted_errors.mean(dim=1, keepdim=True),
                feature_errors.mean(dim=1, keepdim=True)
            ], dim=-1)
        )
        return anomaly_scores.squeeze(), reconstruction, moe_weights, attn_weights