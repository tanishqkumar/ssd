"""
Flash Attention compatibility layer for ROCm.

Provides flash_attn_varlen_func and flash_attn_with_kvcache using either:
1. The flash-attn pip package (if installed)
2. PyTorch scaled_dot_product_attention as fallback
"""
import torch
import torch.nn.functional as F
from typing import Optional

try:
    from flash_attn import flash_attn_varlen_func as _fa_varlen, flash_attn_with_kvcache as _fa_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    if HAS_FLASH_ATTN:
        return _fa_varlen(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale, causal=causal,
            **kwargs,
        )
    # SDPA fallback for variable-length batched attention
    batch_size = cu_seqlens_q.shape[0] - 1
    outputs = []
    for i in range(batch_size):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
        qi = q[q_start:q_end]  # [seq_q, nheads, hdim]
        ki = k[k_start:k_end]  # [seq_k, nkvheads, hdim]
        vi = v[k_start:k_end]
        nheads, nkvheads = qi.shape[1], ki.shape[1]
        if nheads != nkvheads:
            rep = nheads // nkvheads
            ki = ki.repeat_interleave(rep, dim=1)
            vi = vi.repeat_interleave(rep, dim=1)
        qi = qi.transpose(0, 1).unsqueeze(0)  # [1, nheads, seq_q, hdim]
        ki = ki.transpose(0, 1).unsqueeze(0)
        vi = vi.transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(
            qi, ki, vi, scale=softmax_scale, is_causal=causal,
        )
        oi = oi.squeeze(0).transpose(0, 1)  # [seq_q, nheads, hdim]
        outputs.append(oi)
    return torch.cat(outputs, dim=0)


def flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    if HAS_FLASH_ATTN:
        return _fa_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens, block_table=block_table,
            softmax_scale=softmax_scale, causal=causal,
            **kwargs,
        )
    # SDPA fallback for paged KV cache attention
    # q: [batch, seqlen_q, nheads, hdim]
    # k_cache/v_cache: [num_blocks, block_size, nkvheads, hdim] (paged) or [batch, max_kv_len, nkvheads, hdim]
    if q.dim() == 3:
        q = q.unsqueeze(1)  # [batch, 1, nheads, hdim]
    batch_size, seqlen_q, nheads, hdim = q.shape
    nkvheads = k_cache.shape[2]
    block_size = k_cache.shape[1]

    outputs = []
    for i in range(batch_size):
        qi = q[i]  # [seqlen_q, nheads, hdim]
        kv_len = cache_seqlens[i].item() if cache_seqlens is not None else k_cache.shape[1]

        if block_table is not None:
            pages = block_table[i]
            num_pages = (kv_len + block_size - 1) // block_size
            page_ids = pages[:num_pages]
            ki = k_cache[page_ids].reshape(-1, nkvheads, hdim)[:kv_len]
            vi = v_cache[page_ids].reshape(-1, nkvheads, hdim)[:kv_len]
        else:
            ki = k_cache[i, :kv_len]
            vi = v_cache[i, :kv_len]

        if nheads != nkvheads:
            rep = nheads // nkvheads
            ki = ki.repeat_interleave(rep, dim=1)
            vi = vi.repeat_interleave(rep, dim=1)

        qi = qi.transpose(0, 1).unsqueeze(0)  # [1, nheads, seqlen_q, hdim]
        ki = ki.transpose(0, 1).unsqueeze(0)
        vi = vi.transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(
            qi, ki, vi, scale=softmax_scale, is_causal=causal,
        )
        oi = oi.squeeze(0).transpose(0, 1)  # [seqlen_q, nheads, hdim]
        outputs.append(oi)
    return torch.cat(outputs, dim=0).reshape(batch_size, seqlen_q, nheads, hdim)
