# -*- coding: utf-8 -*-
"""
独立的 Memory Attention 模块 - 纯 PyTorch 实现（不依赖 sam2）

这个模块复制了 sam2 中的 RoPE Attention 和 Memory Attention 实现，
可以独立于 sam2 库使用。

只需要 PyTorch，不需要 sam2 库。
"""

import math
import os
from functools import partial
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ==================== RoPE Position Encoding ====================

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq).to(xq.device), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
        else:
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def apply_rotary_enc_v2(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs: int,
):
    if repeat_freqs == 0:
        assert x.shape[-2] == 0
    if x.shape[-2] == 0:
        return x

    B, N_heads, N_tokens, C_per_head = x.shape
    if N_tokens == freqs_cis.shape[0] * repeat_freqs:
        x_rope = x
        x_no_rope = None
    else:
        rope_tokens = freqs_cis.shape[0]
        no_rope_tokens = N_tokens // repeat_freqs - rope_tokens
        x = x.view(B, N_heads, repeat_freqs, N_tokens // repeat_freqs, C_per_head)
        x_rope = x[..., no_rope_tokens:, :].reshape(B, N_heads, -1, C_per_head)
        x_no_rope = x[..., :no_rope_tokens, :].reshape(B, N_heads, -1, C_per_head)

    x_rope = torch.view_as_complex(x_rope.float().reshape(*x_rope.shape[:-1], -1, 2))
    if repeat_freqs > 1:
        x_one_frame = x_rope[..., : x_rope.shape[-2] // repeat_freqs, :]
        freqs_cis = reshape_for_broadcast(freqs_cis, x_one_frame)
    else:
        freqs_cis = reshape_for_broadcast(freqs_cis, x_rope)
    if repeat_freqs > 1:
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), repeat_freqs, 1)
        else:
            freqs_cis = (
                freqs_cis.unsqueeze(2)
                .expand(-1, -1, repeat_freqs, -1, -1)
                .flatten(2, 3)
            )
    x_out = torch.view_as_real(x_rope * freqs_cis).flatten(3)
    x_out = x_out.type_as(x).to(x.device)

    if x_no_rope is not None:
        x_out = x_out.view(B, N_heads, repeat_freqs, -1, C_per_head)
        x_no_rope = x_no_rope.view(B, N_heads, repeat_freqs, -1, C_per_head)
        x_out = torch.cat((x_no_rope, x_out), dim=3).view(
            B, N_heads, N_tokens, C_per_head
        )
    return x_out


# ==================== Attention Layers ====================

class Attention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class RoPEAttention(Attention):
    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(32, 32),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.register_buffer("freqs_cis", freqs_cis)
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        w = h = math.sqrt(q.shape[-2])
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class RoPEAttentionv2(Attention):
    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        q_sizes=(32, 32),
        k_sizes=(32, 32),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        self.register_buffer("freqs_cis_q", self.compute_cis(end_x=q_sizes[0], end_y=q_sizes[1]))
        self.register_buffer("freqs_cis_k", self.compute_cis(end_x=k_sizes[0], end_y=k_sizes[1]))

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        num_k_exclude_rope: int = 0,
        rope_k_repeat: int = -1,
    ) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        q = apply_rotary_enc_v2(q, self.freqs_cis_q, repeat_freqs=1)
        num_k_rope = k.size(-2) - num_k_exclude_rope
        k[:, :, :num_k_rope] = apply_rotary_enc_v2(
            k[:, :, :num_k_rope], self.freqs_cis_k, repeat_freqs=rope_k_repeat
        )

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


# ==================== Memory Attention Layer ====================

class MemoryAttentionLayer(nn.Module):
    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = F.relu if activation == "relu" else F.gelu

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self, tgt, memory, query_pos, pos, num_k_exclude_rope=0, rope_k_repeat=-1
    ):
        kwds = {}
        if rope_k_repeat >= 0:
            assert isinstance(self.cross_attn_image, RoPEAttentionv2)
            kwds["num_k_exclude_rope"] = num_k_exclude_rope
            kwds["rope_k_repeat"] = rope_k_repeat
        elif num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
        rope_k_repeat: int = -1,
    ) -> torch.Tensor:
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(
            tgt, memory, query_pos, pos, num_k_exclude_rope, rope_k_repeat
        )
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


# ==================== Memory Attention ====================

def get_clones(module, N):
    return nn.ModuleList([module for _ in range(N)])


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        num_layers: int,
        batch_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first
        
        # 创建层 - 与 EdgeTAM 配置匹配
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self_attn = RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=(32, 32),
                embedding_dim=d_model,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
            )
            cross_attn = RoPEAttentionv2(
                rope_theta=10000.0,
                q_sizes=(64, 64),
                k_sizes=(16, 16),
                embedding_dim=d_model,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
                kv_in_dim=64,
            )
            layer = MemoryAttentionLayer(
                activation="relu",
                cross_attention=cross_attn,
                d_model=d_model,
                dim_feedforward=2048,
                dropout=0.1,
                pos_enc_at_attn=False,
                pos_enc_at_cross_attn_keys=True,
                pos_enc_at_cross_attn_queries=False,
                self_attention=self_attn,
            )
            self.layers.append(layer)

    def forward(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        num_obj_ptr_tokens: int = 0,
        num_spatial_mem: int = -1,
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]

        assert curr.shape[1] == memory.shape[1]

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}
            if isinstance(layer.cross_attn_image, RoPEAttentionv2):
                kwds["num_k_exclude_rope"] = num_obj_ptr_tokens
                kwds["rope_k_repeat"] = num_spatial_mem

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)

        return normed_output


def build_memory_attention(weights_path: str = None, device: str = "cpu") -> Tuple["MemoryAttention", dict]:
    """构建 Memory Attention 模块并加载权重
    
    Args:
        weights_path: 权重文件路径 (memory_attention.pt)
        device: 设备
        
    Returns:
        mem_attn: MemoryAttention 模块
        extras: 包含 no_mem_embed, no_mem_pos_enc, maskmem_tpos_enc 等
    """
    # 创建模块
    mem_attn = MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        num_layers=2,
        batch_first=True,
    )
    
    extras = {}
    config = {}
    
    # 加载权重
    # 注意: freqs_cis 是在初始化时计算的 buffer，不在 state_dict 中
    # 所以使用 strict=False 来忽略缺失的 buffer
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 只加载可学习参数（不包括 freqs_cis buffer）
        missing, unexpected = mem_attn.load_state_dict(state_dict, strict=False)
        if missing:
            # 过滤掉 freqs_cis，因为它是在初始化时计算的
            missing_params = [k for k in missing if 'freqs_cis' not in k]
            if missing_params:
                print(f"  警告: 缺失参数: {missing_params}")
        
        # 加载额外参数
        if 'extras' in checkpoint:
            extras = checkpoint['extras']
            # 转移到设备
            for k, v in extras.items():
                if isinstance(v, torch.Tensor):
                    extras[k] = v.to(device)
        
        # 加载配置
        if 'config' in checkpoint:
            config = checkpoint['config']
        
        print(f"  加载 Memory Attention 权重: {weights_path}")
    
    mem_attn.to(device)
    mem_attn.eval()
    
    return mem_attn, extras, config


# ==================== 测试 ====================
if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 测试构建和加载
    print("测试 Memory Attention 模块...")
    
    weights_path = "onnx_models/memory_attention.pt"
    if os.path.exists(weights_path):
        mem_attn, extras, config = build_memory_attention(weights_path, device="cpu")
        
        print(f"\n配置:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        print(f"\n额外参数:")
        for k, v in extras.items():
            print(f"  {k}: {v.shape}")
        
        # 测试前向传播
        batch_size = 1
        hw = 4096  # 64x64
        mem_len = 512 + 16  # spatial mem + obj ptr
        d_model = 256
        d_mem = 64
        
        curr = torch.randn(hw, batch_size, d_model)
        memory = torch.randn(mem_len, batch_size, d_mem)
        curr_pos = torch.randn(hw, batch_size, d_model)
        memory_pos = torch.randn(mem_len, batch_size, d_mem)
        
        with torch.no_grad():
            output = mem_attn(
                curr, memory, curr_pos, memory_pos,
                num_obj_ptr_tokens=16,
                num_spatial_mem=1,
            )
        
        print(f"\n前向传播测试:")
        print(f"  输入 curr: {curr.shape}")
        print(f"  输入 memory: {memory.shape}")
        print(f"  输出: {output.shape}")
        print("  测试通过!")
    else:
        print(f"  找不到权重文件: {weights_path}")
        print("  请先运行: python export_memory_attention_pt.py")
