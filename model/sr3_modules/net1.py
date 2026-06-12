import math
from inspect import isfunction

from pytorch_tcn import tcn
# from tcn import TemporalConvNet

import torch
from torch import nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """用于在padding后裁剪多余的长度，确保输出与输入时间长度一致"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]  # 裁掉多余padding

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, input_channels, channel_list, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(channel_list)
        for i in range(num_levels):
            in_channels = input_channels if i == 0 else channel_list[i-1]
            out_channels = channel_list[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # 输入 [B, L, D]
        x = x.transpose(1, 2)  # 转为 [B, D, L]
        y = self.network(x)
        return y.transpose(1, 2)  # 输出 [B, L, C]

# class TrendExpert(nn.Module):
#     def __init__(self, input_dim, seq_len, n_channels=[32, 32]):
#         super().__init__()
#         self.tcn = TemporalConvNet(input_dim, n_channels, kernel_size=3, dropout=0.1)
#         self.fc = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(seq_len * n_channels[-1], input_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):  # [B, L, D]
#         h = self.tcn(x)  # [B, L, C]
#         h = h.flatten(start_dim=1)
#         α = self.fc(h)  # [B, D]
#         return α
# #
# class CorrelationExpert(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim=100, num_heads=4, batch_first=True)
#
#     def forward(self, x):  # x: [B, L, D]
#         x = x.transpose(1, 2)  # → [B, D, L]
#         h, _ = self.attn(x, x, x)  # attention across variables
#         h = h.mean(dim=2)  # → [B, D]
#         α = torch.sigmoid(h)
#         return α
# #
# class SpikeExpert(nn.Module):
#     def __init__(self, input_dim, seq_len):
#         super().__init__()
#         self.conv = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
#         self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
#         self.fc = nn.Sequential(
#             nn.Linear(64 * seq_len, input_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):  # x: [B, L, D]
#         x = x.transpose(1, 2)  # [B, D, L]
#         h = self.conv(x).transpose(1, 2)  # → [B, L, 64]
#         h, _ = self.attn(h, h, h)  # → [B, L, 64]
#         α = self.fc(h.flatten(start_dim=1))  # [B, D]
#         return α


class TrendExpert(nn.Module):
    def __init__(self, input_dim, patch_len, n_channels=[32, 32]):
        super().__init__()
        self.tcn = TemporalConvNet(input_dim, n_channels, kernel_size=3, dropout=0.1)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(20 * n_channels[-1], input_dim),
            nn.Sigmoid()
        )

        # print(f"[TrendExpert] Linear in_features: {patch_len}, out_features: {input_dim}")

    def forward(self, x):  # [B*N, L, D]
        h = self.tcn(x)  # [B*N, L, C]
        h = h.flatten(start_dim=1)
        α = self.fc(h)  # [B*N, D]
        return α


class CorrelationExpert(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=20, num_heads=4, batch_first=True)

    def forward(self, x):  # [B*N, L, D]
        x = x.transpose(1, 2)  # → [B*N, D, L]
        h, _ = self.attn(x, x, x)
        h = h.mean(dim=2)  # [B*N, D]
        α = torch.sigmoid(h)
        return α

class SpikeExpert(nn.Module):
    def __init__(self, input_dim, patch_len):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64 * 20, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):  # [B*N, L, D]
        x = x.transpose(1, 2)  # [B*N, D, L]
        h = self.conv(x).transpose(1, 2)  # [B*N, L, 64]
        h, _ = self.attn(h, h, h)
        α = self.fc(h.flatten(start_dim=1))  # [B*N, D]
        return α
# class SeasonalityExpert(nn.Module):
#     def __init__(self, input_dim, patch_len):
#         super().__init__()
#         self.freq_proj = nn.Linear(20, 64)
#         self.fc = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(64 * input_dim, input_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):  # [B*N, L, D]
#         x = x.transpose(1, 2)  # [B*N, D, L]
#         f = torch.fft.fft(x, dim=-1).abs()  # → [B*N, D, L]
#         f = self.freq_proj(f)  # [B*N, D, 64]
#         h = f.flatten(start_dim=1)  # [B*N, D*64]
#         return self.fc(h)  # [B*N, D]

# class ExpertRouter(nn.Module):
#     def __init__(self, input_dim, seq_len, num_experts=3):
#         super().__init__()
#         self.trend_conv = nn.Conv1d(input_dim, 32, kernel_size=9, padding=4)
#         self.spike_conv = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
#         self.pool = nn.AdaptiveAvgPool1d(1)
#
#         self.mlp = nn.Sequential(
#             nn.Linear(32 * 2 + input_dim * input_dim, 64),  # trend + spike + corr
#             nn.ReLU(),
#             nn.Linear(64, num_experts),
#             nn.Softmax(dim=-1)
#         )
#
#     def forward(self, x):  # x: [B, L, D]
#         B, L, D = x.shape
#         x_ = x.permute(0, 2, 1)  # [B, D, L]
#
#         trend_feat = self.pool(self.trend_conv(x_)).squeeze(-1)  # [B, 32]
#         spike_feat = self.pool(self.spike_conv(x_)).squeeze(-1)  # [B, 32]
#
#         x_centered = x - x.mean(dim=1, keepdim=True)
#         corr = torch.einsum('blc,bld->bcd', x_centered, x_centered) / (L - 1)
#         corr_feat = corr.view(B, -1)  # [B, D*D]
#
#         feat = torch.cat([trend_feat, spike_feat, corr_feat], dim=-1)  # [B, 32+32+D*D]
#         return self.mlp(feat)  # [B, num_experts]


class ExpertRouter(nn.Module):
    def __init__(self, input_dim, patch_len, num_experts=3):
        super().__init__()
        self.trend_conv = nn.Conv1d(input_dim, 32, kernel_size=9, padding=4)
        self.spike_conv = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.mlp = nn.Sequential(
            nn.Linear(32 * 2 + input_dim * input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):  # [B*N, L, D]
        B, L, D = x.shape
        x_ = x.permute(0, 2, 1)  # [B*N, D, L]

        trend_feat = self.pool(self.trend_conv(x_)).squeeze(-1)  # [B*N, 32]
        spike_feat = self.pool(self.spike_conv(x_)).squeeze(-1)  # [B*N, 32]

        x_centered = x - x.mean(dim=1, keepdim=True)
        corr = torch.einsum('blc,bld->bcd', x_centered, x_centered) / (L - 1)
        corr_feat = corr.view(B, -1)  # [B*N, D*D]

        feat = torch.cat([trend_feat, spike_feat, corr_feat], dim=-1)
        return self.mlp(feat)  # [B*N, num_experts]


# class ExpertRouter(nn.Module):
#     def __init__(self, input_dim, patch_len, num_experts=4):  # 默认4个专家
#         super().__init__()
#         self.input_dim = input_dim
#         self.patch_len = patch_len
#         self.num_experts = num_experts
#
#         # Trend path: 提取趋势特征
#         self.trend_conv = nn.Conv1d(input_dim, 32, kernel_size=9, padding=4)
#
#         # Spike path: 检测突变
#         self.spike_conv = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
#
#         # Pooling 压缩特征
#         self.pool = nn.AdaptiveAvgPool1d(1)  # 输出 [B*N, C, 1]
#
#         # Correlation path: 简化为快速计算协方差特征
#         self.corr_proj = nn.Linear(input_dim * input_dim, 64)
#
#         # 可选：Seasonality path（可注释掉）
#         self.use_fft = True
#         if self.use_fft:
#             self.fft_proj = nn.Linear(input_dim * (20 // 2 + 1), 64)
#
#         # Router MLP: 输入 trend/spike/corr/fft 融合特征 → 权重
#         feature_size = 32 + 32 + 64 + (64 if self.use_fft else 0)
#         self.mlp = nn.Sequential(
#             nn.Linear(feature_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_experts),
#             nn.Softmax(dim=-1)
#         )
#
#     def forward(self, x):  # x: [B*N, L, D]
#         B, L, D = x.shape
#         x_perm = x.permute(0, 2, 1)  # [B*N, D, L]
#
#         # Trend and spike features
#         trend_feat = self.pool(self.trend_conv(x_perm)).squeeze(-1)  # [B*N, 32]
#         spike_feat = self.pool(self.spike_conv(x_perm)).squeeze(-1)  # [B*N, 32]
#
#         # Correlation feature: [B*N, D, D]
#         x_centered = x - x.mean(dim=1, keepdim=True)
#         corr = torch.einsum('blc,bld->bcd', x_centered, x_centered) / (L - 1)
#         corr_feat = self.corr_proj(corr.view(B, -1))  # [B*N, 64]
#
#         # Optional: FFT for seasonality
#         if self.use_fft:
#             fft_feat = torch.fft.rfft(x, dim=1).abs()  # [B*N, F, D]
#             fft_feat = fft_feat.flatten(start_dim=1)  # [B*N, D * F]
#             fft_feat = self.fft_proj(fft_feat)  # [B*N, 64]
#             features = torch.cat([trend_feat, spike_feat, corr_feat, fft_feat], dim=-1)
#         else:
#             features = torch.cat([trend_feat, spike_feat, corr_feat], dim=-1)
#
#         weights = self.mlp(features)  # [B*N, num_experts]
#         return weights

# class ExpertPool(nn.Module):
#     def __init__(self, input_dim, seq_len):
#         super().__init__()
#         self.experts = nn.ModuleList([
#             TrendExpert(input_dim, seq_len),
#             SpikeExpert(input_dim, seq_len),
#             CorrelationExpert(input_dim),
#         ])
#         self.router = ExpertRouter(input_dim, seq_len, num_experts=len(self.experts))
#
#     def forward(self, x):  # [B, L, D]
#         expert_weights = self.router(x)  # [B, num_experts]
#         α_list = [expert(x) for expert in self.experts]  # 每个为 [B, D]
#         α_stack = torch.stack(α_list, dim=1)  # [B, num_experts, D]
#
#         # 对每个 expert 加权平均
#         α = torch.einsum('be,bed->bd', expert_weights, α_stack)
#         return α  # [B, D]

class ExpertPool(nn.Module):
    def __init__(self, input_dim, patch_len):
        super().__init__()
        self.experts = nn.ModuleList([
            TrendExpert(input_dim, patch_len),
            SpikeExpert(input_dim, patch_len),
            CorrelationExpert(input_dim),
        ])
        self.router = ExpertRouter(input_dim, patch_len, num_experts=len(self.experts))

    def forward(self, x):  # [B, N, L, D]
        B, N, L, D = x.shape
        x_reshaped = x.view(B * N, L, D)

        expert_weights = self.router(x_reshaped)  # [B*N, num_experts]
        α_list = [expert(x_reshaped) for expert in self.experts]  # list of [B*N, D]
        α_stack = torch.stack(α_list, dim=1)  # [B*N, num_experts, D]

        α = torch.einsum('be,bed->bd', expert_weights, α_stack)  # [B*N, D]
        α = α.view(B, N, D)
        return α  # 每个 patch 的特征异常敏感度

# class ExpertPool(nn.Module):
#     def __init__(self, input_dim, patch_len):
#         super().__init__()
#         self.experts = nn.ModuleList([
#             TrendExpert(input_dim, patch_len),
#             SpikeExpert(input_dim, patch_len),
#             CorrelationExpert(input_dim),
#             SeasonalityExpert(input_dim, patch_len),
#         ])
#         self.router = ExpertRouter(input_dim, patch_len, num_experts=len(self.experts))
#
#     def forward(self, x):  # [B, N, L, D]
#         B, N, L, D = x.shape
#         x_flat = x.view(B * N, L, D)
#         α_list = [expert(x_flat) for expert in self.experts]  # list of [B*N, D]
#         α_stack = torch.stack(α_list, dim=1)  # [B*N, num_experts, D]
#         expert_weights = self.router(x_flat).unsqueeze(-1)  # [B*N, num_experts, 1]
#         α_weighted = (expert_weights * α_stack).sum(dim=1)  # [B*N, D]
#         return α_weighted.view(B, N, D)  # [B, N, D]

