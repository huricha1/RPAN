import math
from functools import partial
from inspect import isfunction

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from model.sr3_modules.net1 import ExpertPool
# from time_train1 import split_into_patches
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 放在代码开头
# 创建一个包含指定层数和头数的 Transformer 编码器
def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers) #output: 形状为 (sequence_length, batch_size, num_heads * head_dim)

def split_into_patches(x):
    """
    x: [b, l, k] → return [b, 3, patch_len, k]
    """
    b, l, k = x.shape
    assert l % 5 == 0
    chunks = torch.chunk(x, chunks=5, dim=1)  # list of 3 tensors, each [b, l//3, k]
    patches = torch.stack(chunks, dim=1)      # [b, 3, patch_len, k]
    return patches

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
        self.linear=nn.Linear(128,38)
        self.sigmoid=nn.Sigmoid()
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 20, 1)
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
        x = x.reshape(B,L,20,K)
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
        # self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

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

        _, cond_dim, _, _ = cond_info.shape  # (B, side_dim, K, L)
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_proj = self.cond_projection(cond_info)  # (B, 2*C, K*L)
        cond_residual, cond_skip = torch.chunk(cond_proj, 2, dim=1)

        # print("strategy emb is")
        # print(strategy_emb)
        # print(strategy_emb.shape)
        # y = x + diffusion_emb
        y = x + diffusion_emb+cond_residual
        y = self.forward_time(y, base_shape)
        # y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        # _, cond_dim, _, _ = cond_info.shape # (B,*,K,L)
        # cond_info = cond_info.reshape(B, cond_dim, K * L)
        # cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        # y = y + cond_info

        # 将张量沿指定维度分割成多个子张量
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y) # (B,2*channel,K*L)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip #除以根号2，这样的归一化操作常用于防止信号在加法后过大，从而保持稳定性

def linear_beta_schedule(timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
def exists(x):
    """
    Check if the input is not None.

    Args:
        x: The input to check.

    Returns:
        bool: True if the input is not None, False otherwise.
    """
    return x is not None
def default(val, d):
    """
    Return the value if it exists, otherwise return the default value.

    Args:
        val: The value to check.
        d: The default value or a callable that returns the default value.

    Returns:
        The value if it exists, otherwise the default value.
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """
    Extracts values from tensor `a` at indices specified by tensor `t` and reshapes the result.
    Args:
        a (torch.Tensor): The input tensor from which values are extracted.
        t (torch.Tensor): The tensor containing indices to extract from `a`.
        x_shape (tuple): The shape of the tensor `x` which determines the final shape of the output.
    Returns:
        torch.Tensor: A tensor containing the extracted values, reshaped to match the shape of `x` except for the first dimension.
    """

    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def identity(t, *args, **kwargs):
    """
    Return the input tensor unchanged.

    Args:
        t: The input tensor.
        *args: Additional arguments (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        The input tensor unchanged.
    """
    return t
# from net import AttentionMoE

class MoEConditionalDiffusion(nn.Module):
    """混合专家引导的条件扩散模型"""

    def __init__(self, config,device,input_dim, seq_len, timesteps,patch_len, d_model):
        super().__init__()
        self.device=device
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["diffusion"]["num_steps"],
            embedding_dim=config["diffusion"]["diffusion_embedding_dim"],
        )
        # input_dim=feature_dim
        self.moe = ExpertPool(input_dim, seq_len)
        # self.diffusion_steps = diffusion_steps
        self.emb_time_dim = config["model"]["timeemb"]
        # self.emb_feature_dim = config["model"]["featureemb"]
        # self.emb_total_dim = self.emb_time_dim + 2*self.emb_feature_dim
        # self.emb_total_dim = self.emb_time_dim +  self.emb_feature_dim
        self.emb_total_dim = self.emb_time_dim
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        self.num_steps = config_diff["num_steps"]

        # self.beta = torch.linspace(
        #     config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
        # ) ** 2
        # self.beta = torch.linspace(
        #     1e-6, 1e-2, self.num_steps
        # )
        # self.alphas = 1. - self.betas
        # self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # self.alpha_hat = 1 - self.beta
        # 计算输入数组的累积乘积 返回数组
        # cumprod函数表示将之前的alpha连乘。这里的self.alpha实际上就是\overline \alpha
        # self.alpha = np.cumprod(self.alpha_hat)
        # self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # self.embed_layer = nn.Embedding(
        #     num_embeddings=input_dim+num_experts, embedding_dim=self.emb_feature_dim
        # )
        # self.embed_layer = nn.Embedding(
        #     num_embeddings=2*input_dim, embedding_dim=self.emb_feature_dim
        # )

        # 条件去噪网络
        self.denoiser = diff_CSDI(config_diff,1)
        self.W_P = nn.Linear(2*input_dim * patch_len, 128)
        # 时间嵌入
        # self.time_embed = nn.Embedding(diffusion_steps, 32)
        # self.time_mlp = nn.Sequential(
        #     nn.Linear(32, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, input_dim)
        # )

        # reconstruct diffusion
        # betas = linear_beta_schedule(timesteps)
        betas=cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        # self.sampling_timesteps = default(
        #     sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps =  timesteps
        assert self.sampling_timesteps <= timesteps

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))


    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def output(self, x, ori,ori_patch,t, padding_masks=None):
        b,c,k ,device = *ori.shape, ori.device
        # 获取MoE权重
        attn_weights = self.moe(ori)
        # moe_weights = expert_weights.mean(dim=1)  # (batch,)
        # moe_weights = expert_weights
        moe_weights = attn_weights
        moe_weights = moe_weights.unsqueeze(1)
        moe_weights = moe_weights.expand(-1, c, -1)
        # [b, 3, patch_len, k]
        moe_weights = split_into_patches(moe_weights)
        # x = split_into_patches(x)
        #
        # noise = torch.randn_like(x)
        #
        # noisy_x = self.q_sample(x_start=x, t=t, noise=noise)  # noise sample
        # 条件去噪
        rec = self.denoise_step(x, t, moe_weights)
        return rec,moe_weights

    def model_predictions(self, x, ori,ori_patch,t, clip_x_start=False, padding_masks=None):

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start,moe_weights = self.output(x, ori,ori_patch,t, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def p_mean_variance(self, x, ori,ori_patch,t, clip_denoised=True):
        _, x_start = self.model_predictions(x, ori,ori_patch,t)
        if clip_denoised:
            x_start.clamp_(-3., 3.)
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x,ori, ori_patch ,t: int, clip_denoised=True, cond_fn=None, model_kwargs=None):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(x=x,ori=ori,ori_patch=ori_patch,t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_series, x_start

    @torch.no_grad()
    def sample(self, x,x_patch):
        device = self.betas.device
        img = torch.randn(x_patch.shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            img, _ = self.p_sample(img, x,x_patch,t)
        return img



    def generate_mts(self, x,x_patch):
        sample_fn = self.sample
        return sample_fn(x,x_patch)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample

        model_out = self.output(x, t, padding_masks)

        train_loss = self.loss_fn(model_out, target, reduction='none')


        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def forward(self, x, **kwargs):
        # b, c, n, device = *x.shape, x.device
        # # assert n == feature_size, f'number of variable must be {feature_size}'
        # t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # return self._train_loss(x_start=x, t=t, **kwargs)

        b, c, n, device = *x.shape, x.device
        # 获取MoE权重
        attn_weights = self.moe(x)
        # moe_weights = expert_weights.mean(dim=1)  # (batch,)
        # moe_weights = expert_weights
        moe_weights = attn_weights
        moe_weights=moe_weights.unsqueeze(1)
        moe_weights=moe_weights.expand(-1,c,-1)
        # [b, 3, patch_len, k]
        moe_weights = split_into_patches(moe_weights)
        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # print(x.dtype)
        x = split_into_patches(x)

        noise = torch.randn_like(x)

        noisy_x = self.q_sample(x_start=x, t=t, noise=noise)  # noise sample
        # 条件去噪
        pred_noise = self.denoise_step(noisy_x, t, moe_weights)

        return pred_noise, noise, moe_weights

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
        # feature_embed = self.embed_layer(
        #     torch.arange(K).to(self.device)
        # )  # (K,emb) 为[0,target_dim)的feature嵌入emd
        # feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)#expand方法是用于扩展张量维度的，可以用来复制张量的形状 （B,L,K,EMB）
        #
        # side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info=time_embed
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

        x = split_into_patches(x)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(x)
        # noise = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        noisy_data = (current_alpha ** 0.5) * x + (1.0 - current_alpha) ** 0.5 * noise
        return noisy_data, noise
        # return noisy_data

    def denoise_step(self, noisy_x, t, moe_weights):
            """条件去噪步骤"""

            t_emb=self.diffusion_embedding(t)
            # 应用MoE权重条件
            # weighted_condition = torch.einsum('btf,b->btf', moe_weights, t_emb)
            # denoiser_input = torch.cat([noisy_x, moe_weights.permute(0,2,1)], dim=-1)
            denoiser_input = torch.cat([noisy_x, moe_weights], dim=-1)
            bs,patch_num,_,_=denoiser_input.shape
            # denoiser_input = noisy_x
            denoiser_input = denoiser_input.reshape(bs, patch_num, -1)  # flatten: [bs, patch_num, nvars * patch_len]
            denoiser_input = self.W_P(denoiser_input)  # [bs, patch_num, d_model]
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

    # def forward(self, x):
    #     """训练前向传播"""
    #     # 获取MoE权重
    #     attn_weights = self.moe(x)
    #     # moe_weights = expert_weights.mean(dim=1)  # (batch,)
    #     # moe_weights = expert_weights
    #     moe_weights = attn_weights
    #     # 随机采样时间步
    #     batch_size = x.size(0)
    #     t = torch.randint(0, self.diffusion_steps, (batch_size,))
    #     # print(x.dtype)
    #     # 前向扩散
    #     noisy_x, noise = self.diffuse(x, t)
    #
    #     # 条件去噪
    #     pred_noise = self.denoise_step(noisy_x, t, moe_weights)
    #
    #     return pred_noise, noise, moe_weights, attn_weights
    #
    # def sample(self, x):
    #     """完整扩散去噪过程（支持三维输入）"""
    #     self.moe.eval()
    #     self.denoiser.eval()
    #     with torch.no_grad():
    #         # 1. 获取MoE权重（处理三维输入）
    #         _, expert_weights, attn_weights = self.moe(x)
    #         # moe_weights = expert_weights
    #         moe_weights = _
    #         # 2. 从纯噪声开始         （保持三维结构）
    #         # noisy_x = torch.randn_like(x)  # (batch_size, timesteps, num_features)
    #
    #         # 3. 完整反向扩散过程
    #         # for t in reversed(range(self.diffusion_steps)):
    #         #     # 当前时间步张量
    #         #     t_tensor = torch.full((x.size(0),), t, device=x.device)  # (batch_size,)
    #         #
    #         #     # 4. 去噪步骤（需适配三维输入）
    #         #     pred_noise = self.denoise_step(
    #         #         noisy_x,  # 直接传入三维
    #         #         t_tensor,
    #         #         moe_weights
    #         #     )  # 输出应保持 (batch_size, timesteps, num_features)
    #         #
    #         #     # 5. 计算扩散参数（扩展维度以支持广播）
    #         #     alpha_t = self.alphas[t].view(-1, 1, 1).to(x.device)
    #         #     alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1).to(x.device)
    #         #     beta_t = self.betas[t].view(-1, 1, 1).to(x.device)
    #         #     sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t).to(x.device)
    #         #     sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t).to(x.device)
    #         #
    #         #     # 6. 更新噪声数据（三维操作）
    #         #     if t > 0:
    #         #         noise = torch.randn_like(noisy_x)
    #         #     else:
    #         #         noise = torch.zeros_like(noisy_x)
    #         #
    #         #     noisy_x = (
    #         #                       sqrt_recip_alpha_t *
    #         #                       (noisy_x - beta_t * pred_noise / sqrt_one_minus_alpha_bar_t)
    #         #               ) + torch.sqrt(beta_t) * noise
    #
    #         current_sample = torch.randn_like(x)
    #
    #         for t in range(self.num_steps - 1, -1, -1):  # 反向迭代
    #             diff_input = current_sample
    #             # diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
    #
    #             t_tensor = torch.full((x.size(0),), t, device=x.device)  # (batch_size,)
    #             # 4. 去噪步骤（需适配三维输入）
    #             predicted = self.denoise_step(
    #                 diff_input,  # 直接传入三维
    #                 t_tensor,
    #                 moe_weights
    #             )  # 输出应保持 (batch_size, timesteps, num_features)
    #             coeff1 = 1 / self.alpha_hat[t] ** 0.5
    #
    #             # 注意一下，这里的alpha_hat以及alpha和DDPM论文当中的alpha是正好相反的。
    #             coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
    #             current_sample = coeff1 * (current_sample - coeff2 * predicted)
    #
    #             if t > 0:
    #                 noise = torch.randn_like(current_sample)
    #                 sigma = (
    #                                 (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
    #                         ) ** 0.5
    #                 current_sample += sigma * noise
    #
    #         return current_sample, moe_weights, attn_weights  # 返回三维重建结果
