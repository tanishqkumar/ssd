import torch
from torch import nn
import triton
import triton.language as tl

# from flash_attn import flash_attn_varlen_func
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
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
    cache_offsets = slot * D + tl.arange(0, D)
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
        self.prefill_wrappers = {}
        self.F = F # async_fan_out
        self.K = K # speculate_k
        self.only_prefill_wrapper = None 

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim) # flatten, no masking required
        k = k.view(-1, self.num_kv_heads, self.head_dim) # we store all seqlen separation info in ctxt at this point, which was 
        v = v.view(-1, self.num_kv_heads, self.head_dim) # built in set_context() in prepare_prefill/decode
        
        k_cache, v_cache = self.k_cache, self.v_cache

        context = get_context()
        if self.k_cache.numel() and self.v_cache.numel():  
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping) 
        
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache, ie. if any of our seqs got a page hit in kvc
                k, v = k_cache, v_cache
            
            k, v = k.view(-1, self.num_kv_heads, self.head_dim), v.view(-1, self.num_kv_heads, self.head_dim)
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q, # uses cumsums of q vs k to mask
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True)
        else:    # decode, incl specdec target verify
            verify_or_glue = (
                self.speculate and context.cu_seqlens_q is not None 
            )
            decode = (
                not verify_or_glue
            )
            tree_decode = (
                decode and (
                    self.speculate and self.draft and self.draft_async)
            )

            if verify_or_glue:
                # assume q = q_varlen is correctly at this point (-1, nh, hd) like in prefill path, no need to reshape
                assert context.context_lens is not None
                o = flash_attn_with_kvcache(q, k_cache, v_cache, 
                                        cache_seqlens=context.context_lens, page_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True, 
                                        cu_seqlens_q=context.cu_seqlens_q, max_seqlen_q=context.max_seqlen_q, 
                                        )

            else: # decode                                                                         
                if tree_decode and not context.is_jit: 
                        if self.only_prefill_wrapper is not None: # async, eager -- no need to capture batchsize-specific kernel wrappers 
                            prefill_wrapper = self.only_prefill_wrapper
                        else:
                            mq_len = self.F * (self.K+1)
                            bs = q.shape[0] // mq_len
                            
                            # Find the smallest FI wrapper >= our bs
                            wrapper_bs = None
                            for available_bs in sorted(self.prefill_wrappers.keys()):
                                if available_bs >= bs:
                                    wrapper_bs = available_bs
                                    break
                            prefill_wrapper = self.prefill_wrappers[wrapper_bs]
                        o = prefill_wrapper.run(q, (self.k_cache, self.v_cache))
                else: # single query decode, sync spec and normal decoding 
                    # [bs, 1, num_heads, head_dim]
                    q = q.unsqueeze(1)
                    o = flash_attn_with_kvcache(q, k_cache, v_cache,
                                                cache_seqlens=context.context_lens, page_table=context.block_tables,
                                                softmax_scale=self.scale, causal=True, 
                                                )
        
        o = o.view(-1, self.num_heads * self.head_dim) # 2d shape expected by qwen/llama fwd pass as MLPs are per-tok
        return o
