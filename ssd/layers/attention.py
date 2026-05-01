import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

_ATTN_BACKEND = None
_on_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _ATTN_BACKEND = "flash_attn"
except ImportError:
    if not _on_rocm:
        try:
            from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
            _ATTN_BACKEND = "sgl_kernel"
        except ImportError:
            from ssd.layers.flash_attn_compat import flash_attn_varlen_func, flash_attn_with_kvcache
            _ATTN_BACKEND = "sdpa_compat"
    else:
        from ssd.layers.flash_attn_compat import flash_attn_varlen_func, flash_attn_with_kvcache
        _ATTN_BACKEND = "sdpa_compat"
from ssd.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot.to(tl.int64) * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        draft: bool = False,
        speculate: bool = False,
        draft_async: bool = False,
        use_eagle: bool = False,
        F: int = 1,
        K: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.draft = draft
        self.speculate = speculate
        self.draft_async = draft_async
        self.use_eagle = use_eagle
        self.prefill_wrappers = {}
        self.F = F # async_fan_out
        self.K = K # speculate_k
        self.only_prefill_wrapper = None

    def _tree_decode_sdpa(self, q: torch.Tensor, context) -> torch.Tensor:
        """SDPA-based tree decode: bypasses AMD FlashInfer BatchPrefill which
        has a JIT kernel bug that produces wrong attention outputs."""
        B = context.context_lens.shape[0]
        MQ_LEN = q.shape[0] // B
        block_size = self.k_cache.shape[1]

        max_kv = int(context.context_lens.max().item())

        positions = torch.arange(max_kv, device=q.device)
        blk_idx = positions // block_size
        pos_in_blk = positions % block_size
        page_ids = context.block_tables[:, blk_idx]
        flat_idx = (page_ids.long() * block_size + pos_in_blk.unsqueeze(0))

        k_flat = self.k_cache.reshape(-1, self.num_kv_heads, self.head_dim)
        v_flat = self.v_cache.reshape(-1, self.num_kv_heads, self.head_dim)

        k_gathered = k_flat[flat_idx]  # [B, max_kv, nkvheads, hdim]
        v_gathered = v_flat[flat_idx]

        rep = self.num_heads // self.num_kv_heads
        if rep > 1:
            k_gathered = k_gathered.repeat_interleave(rep, dim=2)
            v_gathered = v_gathered.repeat_interleave(rep, dim=2)

        q_4d = q.reshape(B, MQ_LEN, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_4d = k_gathered.permute(0, 2, 1, 3)
        v_4d = v_gathered.permute(0, 2, 1, 3)

        mask_4d = context.custom_mask.unsqueeze(1)  # [B, 1, MQ, max_kv]

        o = F.scaled_dot_product_attention(q_4d, k_4d, v_4d,
                                           attn_mask=mask_4d,
                                           scale=self.scale)
        return o.permute(0, 2, 1, 3).reshape(B * MQ_LEN, self.num_heads, self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        k_cache, v_cache = self.k_cache, self.v_cache

        context = get_context()
        if self.k_cache.numel() and self.v_cache.numel():
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache

            k, v = k.view(-1, self.num_kv_heads, self.head_dim), v.view(-1, self.num_kv_heads, self.head_dim)
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True)
        else:
            # verify/glue decode: multi-query with cu_seqlens_q (K+1 or variable per seq)
            verify_or_glue = (
                self.speculate and context.cu_seqlens_q is not None
            )
            decode = not verify_or_glue
            tree_decode = (
                decode and self.speculate and self.draft and self.draft_async
                and not context.is_jit
            )

            if verify_or_glue:
                assert context.context_lens is not None
                if _ATTN_BACKEND == "sgl_kernel":
                    o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                            cache_seqlens=context.context_lens, page_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True,
                                            cu_seqlens_q=context.cu_seqlens_q, max_seqlen_q=context.max_seqlen_q,
                                            )
                else:
                    batch_size = context.context_lens.shape[0]
                    q = q.reshape(batch_size, context.max_seqlen_q, self.num_heads, self.head_dim)
                    o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True,
                                            )

            elif tree_decode:
                if getattr(context, 'custom_mask', None) is not None:
                    o = self._tree_decode_sdpa(q, context)
                elif self.only_prefill_wrapper is not None:
                    prefill_wrapper = self.only_prefill_wrapper
                    o = prefill_wrapper.run(q, (self.k_cache, self.v_cache))
                else:
                    mq_len = self.F * (self.K+1)
                    bs = q.shape[0] // mq_len
                    wrapper_bs = None
                    for available_bs in sorted(self.prefill_wrappers.keys()):
                        if available_bs >= bs:
                            wrapper_bs = available_bs
                            break
                    prefill_wrapper = self.prefill_wrappers[wrapper_bs]
                    o = prefill_wrapper.run(q, (self.k_cache, self.v_cache))
            else: # single query decode
                q = q.unsqueeze(1)
                if _ATTN_BACKEND == "sgl_kernel":
                    o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                                cache_seqlens=context.context_lens, page_table=context.block_tables,
                                                softmax_scale=self.scale, causal=True,
                                                )
                else:
                    o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                                cache_seqlens=context.context_lens, block_table=context.block_tables,
                                                softmax_scale=self.scale, causal=True,
                                                )

        o = o.view(-1, self.num_heads * self.head_dim)
        return o
