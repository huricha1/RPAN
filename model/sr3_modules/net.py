import math
from inspect import isfunction

import torch
from torch import nn
import torch.nn.functional as F

class FeatureAttention(nn.Module):
    """特征级注意力机制"""

    def __init__(self, input_dim):
        super().__init__()
        # window size
        input_dim=30
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x形状: (batch, timesteps, features) 应该改为 (batch, features,timesteps)
        x = x.transpose(1, 2)  # 从 (B,T,F) -> (B,F,T)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(x.size(-1)))
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 应用注意力
        weighted_features = torch.matmul(attn_weights, V)
        return weighted_features, attn_weights


class Expert(nn.Module):
    """专业化特征专家"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)


class AttentionMoE(nn.Module):
    """注意力门控的混合专家模型"""

    def __init__(self,input_dim, num_experts, hidden_dim=64):
        super().__init__()
        self.feature_attention = FeatureAttention(input_dim)
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # 特征级注意力
        attn_features, attn_weights = self.feature_attention(x)

        # 聚合时间信息
        time_aggregated = attn_features.transpose(1, 2)  # (batch, time,features)

        # 门控网络输入: [原始特征聚合, 注意力特征聚合]
        gate_input = torch.cat([x, time_aggregated], dim=-1)
        expert_weights = self.gate(gate_input).permute(0,2,1)  # (batch, num_experts,time)

        # 专家处理
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(time_aggregated))

        # 加权重构
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, num_experts,time, features)
        weighted_output = torch.einsum('bet,betf->btf', expert_weights, expert_outputs)

        return weighted_output, expert_weights, attn_weights
