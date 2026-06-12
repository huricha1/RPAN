import math
from functools import partial
from inspect import isfunction

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from model.sr3_modules.net import AttentionMoE
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 放在代码开头
# 创建一个包含指定层数和头数的 Transformer 编码器
def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers) #output: 形状为 (sequence_length, batch_size, num_heads * head_dim)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size) #对输入数据的最后一维进行一维卷积
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        # projection_dim=none
        if projection_dim is None:
            projection_dim = embedding_dim
        # 注册一个不会被视为模型参数的张量。缓冲区的值不会在优化过程中更新
        # persistent=False: 当你调用 torch.save 保存模型时，这个缓冲区的值不会被存储，而当你加载模型时，它也不会被恢复
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x
    # 创建了一个位置编码矩阵，其中每个时间步的嵌入由正弦和余弦函数计算得出，帮助模型捕捉输入序列的相对和绝对位置
    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1) 创建一个从 0 到 num_steps - 1 的张量
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim) 表示每个时间步在不同频率下的值。
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=1):
        super().__init__()
        # self.moe = AttentionMoE(input_dim, num_experts)
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        # self.strategy_embedding = nn.Embedding(
        #     2, config['diffusion_embedding_dim']
        # )
        self.linear=nn.Linear(19*2,19)
        self.sigmoid=nn.Sigmoid()
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight) # 将给定张量的所有元素初始化为零
        # 创建一个指定数量的残差块，并将这些块存储在 self.residual_layers 中
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):

        B, inputdim, K, L = x.shape #(B,1,K,L)
        # x是noisy data
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        # print("strategy type is")
        # print(strategy_type)

        # strategy_emb = self.strategy_embedding(strategy_type)
        # print("strategy emb is")
        # print(strategy_emb.shape)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)
        # torch.stack：将多个张量按照指定的维度进行堆叠，维度增加  torch.sum：求和后维度减少
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        # x = x.reshape(B, K, L)
        x = x.reshape(B,L,K)
        x=self.linear(x)
        # x=self.sigmoid(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        # self.strategy_projection = nn.Linear(diffusion_embedding_dim, channels)

        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0) #（b*k,c,l）
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        # strategy_emb = self.strategy_projection(strategy_emb).unsqueeze(-1)

        # print("strategy emb is")
        # print(strategy_emb)
        # print(strategy_emb.shape)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape # (B,*,K,L)
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info
        # 将张量沿指定维度分割成多个子张量
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y) # (B,2*channel,K*L)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip #除以根号2，这样的归一化操作常用于防止信号在加法后过大，从而保持稳定性

# from net import AttentionMoE

class MoEConditionalDiffusion(nn.Module):
    """混合专家引导的条件扩散模型"""

    def __init__(self, config,device,input_dim, num_experts, diffusion_steps=100):
        super().__init__()
        self.device=device
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["diffusion"]["num_steps"],
            embedding_dim=config["diffusion"]["diffusion_embedding_dim"],
        )
        # input_dim=feature_dim
        self.moe = AttentionMoE(input_dim, num_experts)
        self.diffusion_steps = diffusion_steps
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        # self.emb_total_dim = self.emb_time_dim + 2*self.emb_feature_dim
        self.emb_total_dim = self.emb_time_dim +  self.emb_feature_dim

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        self.num_steps = config_diff["num_steps"]

        # self.beta = torch.linspace(
        #     config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
        # ) ** 2
        self.beta = torch.linspace(
            1e-6, 1e-2, self.num_steps
        )
        # self.alphas = 1. - self.betas
        # self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.alpha_hat = 1 - self.beta
        # 计算输入数组的累积乘积 返回数组
        # cumprod函数表示将之前的alpha连乘。这里的self.alpha实际上就是\overline \alpha
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

        # self.embed_layer = nn.Embedding(
        #     num_embeddings=input_dim+num_experts, embedding_dim=self.emb_feature_dim
        # )
        self.embed_layer = nn.Embedding(
            num_embeddings=2*input_dim, embedding_dim=self.emb_feature_dim
        )

        # 条件去噪网络
        self.denoiser = diff_CSDI(config_diff,1)
        # 时间嵌入
        # self.time_embed = nn.Embedding(diffusion_steps, 32)
        # self.time_mlp = nn.Sequential(
        #     nn.Linear(32, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, input_dim)
        # )
    def time_embedding(self, B,L, d_model=128):
        pe = torch.zeros(B, L, d_model).to(self.device)
        position = torch.from_numpy(np.tile(np.arange(L), (B, 1))).unsqueeze(2).to(self.device)
        # 生成一系列从 0 到 d_model 的偶数（0, 2, 4, ...），然后将这些值标准化到 (0, 1) 的范围，最后计算它们的 10000 的幂的倒数
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        # pe 的偶数维度（0, 2, 4, ...）被赋值为对应位置的正弦值，而奇数维度（1, 3, 5, ...）被赋值为余弦值
        pe[:, :, 0::2] = torch.sin(position * div_term) #(batch_size, sequence_length, len(div_term))
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe # b*l*d_model
    # observed_tp=(b,l)
    def get_side_info(self, cond_mask):
        cond_mask=cond_mask.permute(0,2,1)
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(B, L,self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1) #unsqueeze(2)：在张量的第 3 维添加一个新的维度(batch_size, sequence_length, 1, d_model)
        feature_embed = self.embed_layer(
            torch.arange(K).to(self.device)
        )  # (K,emb) 为[0,target_dim)的feature嵌入emd
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)#expand方法是用于扩展张量维度的，可以用来复制张量的形状 （B,L,K,EMB）

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        # if self.is_unconditional == False:
        #     side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
        #     side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info
    def _create_beta_schedule(self):
        """创建扩散噪声调度"""
        # scale = 1000 / self.diffusion_steps
        # scale=1
        # beta_start = scale * 0.0001
        # beta_end = scale * 0.02
        # return torch.linspace(beta_start, beta_end, self.diffusion_steps)

        return self.beta

    def diffuse(self, x, t):
        """前向扩散过程"""
        # sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])
        # sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t])
        #
        # noise = torch.randn_like(x)
        # noisy_x = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        # return noisy_x, noise

        """
            前向扩散过程: x shape = (batch, seq_len, input_dim), t shape = (batch,)
            """

        # batch_size, seq_len, input_dim = x.shape
        # device = x.device

        # sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(batch_size, 1, 1).to(device)
        # sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(batch_size, 1, 1).to(device)
        #
        # noise = torch.randn_like(x)
        # noisy_x = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        # return noisy_x, noise


        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(x)
        # noise = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        noisy_data = (current_alpha ** 0.5) * x + (1.0 - current_alpha) ** 0.5 * noise
        return noisy_data, noise
        # return noisy_data

    def denoise_step(self, noisy_x, t, moe_weights):
            """条件去噪步骤"""
            # 时间嵌入

            t_emb=self.diffusion_embedding(t)
            # 应用MoE权重条件
            # weighted_condition = torch.einsum('btf,b->btf', moe_weights, t_emb)
            # denoiser_input = torch.cat([noisy_x, moe_weights.permute(0,2,1)], dim=-1)
            denoiser_input = torch.cat([noisy_x, moe_weights], dim=-1)
            # denoiser_input = noisy_x
            get_side = self.get_side_info(denoiser_input)
            # print(get_side)
            denoiser_input=denoiser_input.permute(0, 2, 1).unsqueeze(1)

            # 预测噪声
            pred_noise = self.denoiser(denoiser_input,get_side,t)
            return pred_noise

            # """
            #     条件去噪步骤: noisy_x shape = (batch, seq_len, input_dim), moe_weights shape = (batch,)
            #     """
            # batch_size, seq_len, input_dim = noisy_x.shape
            # device = noisy_x.device
            #
            # t_emb = self.diffusion_embedding(t)
            # # 应用 MoE 权重条件
            # cond = moe_weights.unsqueeze(-1) * t_emb  # (batch, input_dim)
            # cond = cond.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, input_dim)
            #
            # denoiser_input = torch.cat([noisy_x, cond], dim=-1)  # (batch, seq_len, input_dim * 2)
            #
            # # 处理序列每个时间步的去噪
            # pred_noise = self.denoiser(denoiser_input)  # (batch, seq_len, input_dim)
            # return pred_noise

    def forward(self, x):
        """训练前向传播"""
        # 获取MoE权重
        moe_output, expert_weights, attn_weights = self.moe(x)
        # moe_weights = expert_weights.mean(dim=1)  # (batch,)
        # moe_weights = expert_weights
        moe_weights = moe_output
        # 随机采样时间步
        batch_size = x.size(0)
        t = torch.randint(0, self.diffusion_steps, (batch_size,))
        # print(x.dtype)
        # 前向扩散
        noisy_x, noise = self.diffuse(x, t)

        # 条件去噪
        pred_noise = self.denoise_step(noisy_x, t, moe_weights)

        return pred_noise, noise, moe_weights, attn_weights

    def sample(self, x):
        """完整扩散去噪过程（支持三维输入）"""
        self.moe.eval()
        self.denoiser.eval()
        with torch.no_grad():
            # 1. 获取MoE权重（处理三维输入）
            _, expert_weights, attn_weights = self.moe(x)
            # moe_weights = expert_weights
            moe_weights = _
            # 2. 从纯噪声开始         （保持三维结构）
            # noisy_x = torch.randn_like(x)  # (batch_size, timesteps, num_features)

            # 3. 完整反向扩散过程
            # for t in reversed(range(self.diffusion_steps)):
            #     # 当前时间步张量
            #     t_tensor = torch.full((x.size(0),), t, device=x.device)  # (batch_size,)
            #
            #     # 4. 去噪步骤（需适配三维输入）
            #     pred_noise = self.denoise_step(
            #         noisy_x,  # 直接传入三维
            #         t_tensor,
            #         moe_weights
            #     )  # 输出应保持 (batch_size, timesteps, num_features)
            #
            #     # 5. 计算扩散参数（扩展维度以支持广播）
            #     alpha_t = self.alphas[t].view(-1, 1, 1).to(x.device)
            #     alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1).to(x.device)
            #     beta_t = self.betas[t].view(-1, 1, 1).to(x.device)
            #     sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t).to(x.device)
            #     sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t).to(x.device)
            #
            #     # 6. 更新噪声数据（三维操作）
            #     if t > 0:
            #         noise = torch.randn_like(noisy_x)
            #     else:
            #         noise = torch.zeros_like(noisy_x)
            #
            #     noisy_x = (
            #                       sqrt_recip_alpha_t *
            #                       (noisy_x - beta_t * pred_noise / sqrt_one_minus_alpha_bar_t)
            #               ) + torch.sqrt(beta_t) * noise

            current_sample = torch.randn_like(x)

            for t in range(self.num_steps - 1, -1, -1):  # 反向迭代
                diff_input = current_sample
                # diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)

                t_tensor = torch.full((x.size(0),), t, device=x.device)  # (batch_size,)
                # 4. 去噪步骤（需适配三维输入）
                predicted = self.denoise_step(
                    diff_input,  # 直接传入三维
                    t_tensor,
                    moe_weights
                )  # 输出应保持 (batch_size, timesteps, num_features)
                coeff1 = 1 / self.alpha_hat[t] ** 0.5

                # 注意一下，这里的alpha_hat以及alpha和DDPM论文当中的alpha是正好相反的。
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise

            return current_sample, moe_weights, attn_weights  # 返回三维重建结果
