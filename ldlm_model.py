"""The LDLM diffusion model."""

import math

from einops import rearrange
import k_diffusion as K
import torch
import torch._dynamo
from torch import nn
from torch.nn import functional as F

torch._dynamo.config.suppress_errors = True


# Kernels

def rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    *shape, d, r = x.shape
    return x.view(*shape, d * r)


def _apply_rotary_emb(freqs, t, start_index=0, scale=1.0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)


def _geglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


def _rms_norm(x, scale, eps):
    dtype = torch.promote_types(x.dtype, torch.float32)
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


try:
    apply_rotary_emb = torch.compile(_apply_rotary_emb)
    geglu = torch.compile(_geglu)
    rms_norm = torch.compile(_rms_norm)
except RuntimeError:
    apply_rotary_emb = _apply_rotary_emb
    geglu = _geglu
    rms_norm = _rms_norm


# Rotary Position Embedding


def freqs_lang(theta=10000.0):
    def init(shape):
        dim = shape[-1] * 2
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)] / dim))
        return freqs.log().expand(shape)

    return init


class RoPE(nn.Module):
    def __init__(self, dim, n_heads, start_index=0, freqs_init=freqs_lang(theta=10000.0)):
        super().__init__()
        self.start_index = start_index
        self.freqs = nn.Parameter(freqs_init((n_heads, dim // 2)))

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 2}, n_heads={self.freqs.shape[0]}, start_index={self.start_index}"

    def get_freqs(self, pos):
        freqs = self.freqs.exp().repeat_interleave(2, dim=-1)
        # pos is (..., seq_len)
        # freqs is (n_heads, d_head)
        # their product is (..., n_heads, seq_len, d_head)
        return pos[..., None, :, None] * freqs[:, None, :]

    def forward(self, x, pos):
        freqs = self.get_freqs(pos)
        return apply_rotary_emb(freqs, x, self.start_index)


# Layers

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def round_to_power_of_two(x, tol):
    approxs = []
    for i in range(math.ceil(math.log2(x))):
        mult = 2**i
        approxs.append(round(x / mult) * mult)
    for approx in reversed(approxs):
        error = abs((approx - x) / x)
        if error <= tol:
            return approx
    return approxs[0]


class GEGLU(nn.Module):
    def forward(self, x):
        return geglu(x)


class RMSNorm(nn.Module):
    def __init__(self, param_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(param_shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class QKNorm(nn.Module):
    def __init__(self, n_heads, eps=1e-6, scale_init=10.0, max_scale=100.0):
        super().__init__()
        self.eps = eps
        self.max_scale = math.log(max_scale)
        self.scale = nn.Parameter(torch.full((n_heads,), math.log(scale_init)))
        self.proj_()

    def extra_repr(self):
        return f"n_heads={self.scale.shape[0]}, eps={self.eps}"

    @torch.no_grad()
    def proj_(self):
        """Modify the scale in-place so it doesn't get "stuck" with zero gradient if it's clamped
        to the max value."""
        self.scale.clamp_(max=self.max_scale)

    def forward(self, x):
        self.proj_()
        scale = torch.exp(0.5 * self.scale - 0.25 * math.log(x.shape[-1]))
        return rms_norm(x, scale[:, None, None], self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(cond_features, features, bias=False))

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond) + 1, self.eps)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, d_model)
        self.up_proj = nn.Linear(d_model, d_ff * 2, bias=False)
        self.act = GEGLU()
        self.dropout = nn.Dropout(dropout)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x, cond):
        x = self.norm(x, cond)
        x = self.up_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, dropout=0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.qk_norm = QKNorm(self.n_heads)
        self.pos_emb = RoPE(d_head, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))

    def forward(self, x, pos, attn_mask, cond):
        x = self.norm(x, cond)
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = rearrange(q, "n l (h e) -> n h l e", e=self.d_head)
        k = rearrange(k, "n l (h e) -> n h l e", e=self.d_head)
        v = rearrange(v, "n l (h e) -> n h l e", e=self.d_head)
        q = self.pos_emb(self.qk_norm(q), pos)
        k = self.pos_emb(self.qk_norm(k), pos)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask)
        x = rearrange(x, "n h l e -> n l (h e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, d_head, dropout=0.0):
        super().__init__()
        self.attn = SelfAttentionBlock(d_model, d_head, dropout=dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout=dropout)

    def forward(self, x, pos, attn_mask, cond):
        x = x + self.attn(x, pos, attn_mask, cond)
        x = x + self.ff(x, cond)
        return x


class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = nn.Linear(d_model, d_ff * 2, bias=False)
        self.act = GEGLU()
        self.dropout = nn.Dropout(dropout)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x):
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.out_norm(x)
        return x


# Main model


def padding_mask_to_attn_mask(padding_mask):
    n, s = padding_mask.shape
    eye = torch.eye(s, device=padding_mask.device)
    base_mask = torch.ones([n, s, s], device=padding_mask.device)
    mask = torch.maximum(base_mask * padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2), eye)
    return mask.bool().unsqueeze(1)


class LDLM(nn.Module):
    def __init__(self, n_layers, d_model, z_dim, ctx_len, d_ff=None, dropout=0.0, sigma_data=1.0):
        super().__init__()
        self.sigma_data = sigma_data
        d_ff = d_ff or round_to_power_of_two(d_model * 8 / 3, 0.05)

        self.time_emb = K.layers.FourierFeatures(1, d_model)
        self.time_in_proj = nn.Linear(d_model, d_model, bias=False)
        self.mapping = MappingNetwork(1, d_model, d_ff, dropout=dropout)

        self.z_in_proj = nn.Linear(z_dim, d_model, bias=True)
        self.z_prev_in_proj = nn.Linear(z_dim, d_model, bias=True)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_ff, 64, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)
        self.out_proj = zero_init(nn.Linear(d_model, z_dim, bias=False))

        self.config = {"n_layers": n_layers, "d_model": d_model, "z_dim": z_dim, "ctx_len": ctx_len, "d_ff": d_ff, "sigma_data": sigma_data}

    def param_groups(self, base_lr=1e-4, mapping_lr_scale=1 / 3):
        mapping_names = []
        wd_names = []
        for name, _ in self.named_parameters():
            if name.startswith("mapping"):
                mapping_names.append(name)
            if "norm.linear" in name:
                mapping_names.append(name)
        for name, _ in self.named_parameters():
            if name.startswith("mapping") or name.startswith("blocks"):
                if name.endswith(".weight"):
                    wd_names.append(name)
        wd, no_wd, mapping_wd, mapping_no_wd = [], [], [], []
        for name, param in self.named_parameters():
            if name in wd_names and name not in mapping_names:
                wd.append(param)
            elif name not in wd_names and name not in mapping_names:
                no_wd.append(param)
            elif name in wd_names and name in mapping_names:
                mapping_wd.append(param)
            else:
                mapping_no_wd.append(param)
        groups = [
            {"params": wd, "lr": base_lr},
            {"params": no_wd, "lr": base_lr, "weight_decay": 0.0},
            {"params": mapping_wd, "lr": base_lr * mapping_lr_scale},
            {"params": mapping_no_wd, "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0}
        ]
        return groups

    def forward(self, z, sigma, z_prev, padding_mask):
        # Mapping network
        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        cond = self.mapping(time_emb).unsqueeze(-2)

        # Input embedding
        z = self.z_in_proj(z[:, None])
        z_prev = self.z_prev_in_proj(z_prev / self.sigma_data)

        # Assemble inputs
        x = torch.cat((z, z_prev.flip(-2)), dim=-2)  # z_prev is padded on the left
        allow_attend = padding_mask.new_ones([z.shape[0], 1])
        padding_mask = torch.cat((allow_attend, padding_mask.flip(-1)), dim=-1)
        pos = torch.arange(x.shape[-2], device=x.device)
        attn_mask = padding_mask_to_attn_mask(padding_mask)

        # Transformer
        for block in self.blocks:
            x = block(x, pos, attn_mask, cond)

        # Output embedding
        x = x[:, 0]
        x = self.out_norm(x)
        x = self.out_proj(x)
        return x


# k-diffusion wrapper


class X0Denoiser(nn.Module):
    def __init__(self, inner_model, sigma_data=1.):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data

    def loss(self, input, noise, sigma, **kwargs):
        noised_input = input + noise * K.utils.append_dims(sigma, input.ndim)
        denoised = self(noised_input, sigma, **kwargs)
        c_weight = 1 / (sigma ** 2 + self.sigma_data ** 2)  # soft min-snr weighting
        return (denoised - input).pow(2).flatten(1).mean(1) * c_weight

    def forward(self, input, sigma, **kwargs):
        sigma_ = K.utils.append_dims(sigma, input.ndim)
        c_in = 1 / (sigma_ ** 2 + self.sigma_data ** 2) ** 0.5
        return self.inner_model(input * c_in, sigma, **kwargs) * self.sigma_data
