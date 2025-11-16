import torch
from ssd.utils.async_helpers.async_spec_helpers import get_forked_recovery_tokens_from_logits

def prepare_last_consistency_test(
    branch_bt: torch.Tensor,
    N: int,
    kv_cache: torch.Tensor,
    speculate_k: int,
    async_fan_out: int,
    verbose: bool = False,
):
    """
    Stateless version of prepare_last_consistency_check.
    Checks that the last block of the branch_bt is consistent with the kv_cache.
    Args:
        branch_bt: [N, M] tensor of block ids.
        N: number of forks (rows in branch_bt).
        kv_cache: [2, num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim] tensor.
        speculate_k: int, number of speculative steps.
        async_fan_out: int, number of async fan out branches.
        verbose: bool, whether to print debug info.
    Raises:
        AssertionError if KV cache blocks in a group are not identical.
    """
    group_size = (speculate_k + 1) * async_fan_out
    assert N % group_size == 0, (
        f"ERROR in prepare_last_consistency_check_stateless: N={N} is not divisible by group_size={group_size}"
    )
    B = N // group_size

    # [N] - first block id from each row
    first_block_ids = branch_bt[:, 0]

    for group_idx in range(B):
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size
        # [group_size]
        group_block_ids = first_block_ids[start_idx:end_idx]
        unique_blocks_in_group = torch.unique(group_block_ids)

        if len(unique_blocks_in_group) > 1:
            # Get the kv cache data for the first unique block in this group as reference
            ref_block_id = unique_blocks_in_group[0].item()
            # [num_layers, block_size, num_kv_heads, head_dim]
            ref_k_cache = kv_cache[0, :, ref_block_id]
            ref_v_cache = kv_cache[1, :, ref_block_id]

            # Check all other unique blocks in this group against the reference
            for block_id in unique_blocks_in_group[1:]:
                block_id = block_id.item()
                k_cache = kv_cache[0, :, block_id]
                v_cache = kv_cache[1, :, block_id]

                k_equal = torch.equal(ref_k_cache, k_cache)
                v_equal = torch.equal(ref_v_cache, v_cache)

                assert k_equal and v_equal, (
                    f"ERROR in prepare_glue_decode_ctxt: KV cache mismatch between first blocks in group {group_idx}! "
                    f"Reference block {ref_block_id} vs block {block_id}. "
                    f"K cache equal: {k_equal}, V cache equal: {v_equal}"
                )

    if verbose and N > 1:
        print(
            f"[DRAFT NEW KV CACHE BLOCKS EQUALITY SANITY CHECK 2] KV cache consistency check passed", flush=True)


def logits_out_glue_decode_sanity_test(
    out_logits,
    glue_decode_logits,
    cache_hits,
    make_branch_bt_args,
    speculate_k,
    async_fan_out,
    vocab_size,
    get_forked_recovery_tokens_from_logits_fn=get_forked_recovery_tokens_from_logits,
):
    """
        Stateless sanity check for glue decode logits vs out logits.

        Args:
            out_logits: torch.Tensor, [N, K+1, V]
            glue_decode_logits: torch.Tensor, [N, K+1, V]
            cache_hits: torch.Tensor, [B]
            make_branch_bt_args: dict, must contain "b_flat"
            speculate_k: int, K
            async_fan_out: int, F
            vocab_size: int, V
            get_forked_recovery_tokens_from_logits_fn: function to get rec_flat
        """
    out_logits_up = out_logits[make_branch_bt_args["b_flat"]]
    B = cache_hits.shape[0]

    rec_flat_up = get_forked_recovery_tokens_from_logits_fn(
        out_logits_up, B, speculate_k, async_fan_out)
    print(
        f"[OUT_LOGITS_UP FORK SANITY CHECK 4] rec_flat (forked recovery tokens) from out_logits_up in _build_tree_batch: {rec_flat_up.tolist()}", flush=True)

    K = speculate_k
    F = async_fan_out

    # Both logits are already [N, K+1, V] where N = B(K+1)F

    glue = glue_decode_logits.reshape(B, K+1, F, K+1, vocab_size)
    cached = out_logits_up.reshape(B, K+1, F, K+1, vocab_size)

    any_mismatch = False
    for b in range(B):
        if cache_hits[b].item():
            for k in range(K+1):
                for i in range(k+1):
                    g = glue[b, k, 0, i]
                    c = cached[b, k, 0, i]
                    if not torch.allclose(g, c, rtol=1e-5, atol=1e-6):
                        any_mismatch = True
                        diff = (g - c).abs()
                        print(f"[GLUE vs OUT] MISMATCH b={b} k={k} i={i}: "
                              f"max={diff.max().item():.6e} mean={diff.mean().item():.6e}", flush=True)
                    elif __debug__:
                        print(
                            f"[GLUE vs OUT] match b={b} k={k} i={i}", flush=True)
                # Print top-5 token ids for quick qualitative check
                tops_g = torch.topk(glue[b, k, 0, min(k, K), :], 5).indices.tolist()
                tops_c = torch.topk(cached[b, k, 0, min(k, K), :], 5).indices.tolist()
                print(f"[GLUE vs OUT] b={b} k={k} top1 glue={tops_g[0]} cached={tops_c[0]}", flush=True)
    return any_mismatch
    
    # end sanity check for glue decode logits


def logits_alignment_sanity_test(relevant_logits_reshaped, B, K, F):
    """
    Stateless test function to check that for a given (b, k) pair, the relevant_logits
    are identical across all F branches, and that logits for different (b, k) pairs
    are not identical (except for (0, 0) vs (0, 0)).
    """
    all_pass = True

    for b in range(B):
        for k in range(K + 1):
            logits_at_bk = relevant_logits_reshaped[b, k]  # [F, V]
            # Check all F branches have identical logits
            for f in range(1, F):
                if not torch.allclose(logits_at_bk[0], logits_at_bk[f], atol=1e-6):
                    all_pass = False
            # Check that logits for this (b, k) f=0 are not equal to reference (0, 0) f=0
            if (b, k) != (0, 0):
                ref_logits = relevant_logits_reshaped[0, 0, 0]  # [V]
                if torch.allclose(logits_at_bk[0], ref_logits, atol=1e-6):
                    all_pass = False

    if all_pass:
        print(f'[get_forked_recovery_tokens_from_logits] logits alignment check PASSED :)', flush=True)
    else:
        print(f'[get_forked_recovery_tokens_from_logits] logits alignment check FAILED :(', flush=True)


# self.kv_cache = torch.zeros(
#     2,
#     hf_config.num_hidden_layers,
#     config.num_kvcache_blocks,
#     self.block_size,
#     num_kv_heads,
#     hf_config.head_dim,
# )

# want all slots up to context_len using seq.draft_block_table 
def get_conditioning_tensor_test(kv_cache, block_table, context_len): # use branch_bt and pass in context_len manually to see if our winning fork used the same conditioning tensor when tree decoding as the verifier when verifying
    block_size = kv_cache.shape[3]
    num_full_blocks = context_len // block_size
    len_final_block = context_len % block_size
    
    # Handle edge case where context_len is 0
    if context_len == 0:
        # Return empty tensor with correct shape
        return torch.empty((2, kv_cache.shape[1], 0, kv_cache.shape[4], kv_cache.shape[5]), 
                          dtype=kv_cache.dtype, device=kv_cache.device)
    
    # get block ids up to context len 
    full_block_ids = block_table[:num_full_blocks]
    
    if num_full_blocks > 0:
        # get all slots for all but last block 
        full_block_slots = kv_cache[:, :, full_block_ids, :, :, :]
        # Reshape to concatenate along sequence dimension
        full_block_slots = full_block_slots.reshape(2, kv_cache.shape[1], -1, kv_cache.shape[4], kv_cache.shape[5])
    else:
        full_block_slots = None
    
    # Handle partial block only if there are remaining tokens
    if len_final_block > 0:
        partial_block_id = block_table[num_full_blocks]
        # get partial slots for last block 
        partial_block_slots = kv_cache[:, :, partial_block_id, :len_final_block, :, :]
    else:
        partial_block_slots = None
    
    # concat all slots 
    if full_block_slots is not None and partial_block_slots is not None:
        conditioning_tensor = torch.cat([full_block_slots, partial_block_slots], dim=2)
    elif full_block_slots is not None:
        conditioning_tensor = full_block_slots
    elif partial_block_slots is not None:
        conditioning_tensor = partial_block_slots
    else:
        # This shouldn't happen given the context_len > 0 check above
        conditioning_tensor = torch.empty((2, kv_cache.shape[1], 0, kv_cache.shape[4], kv_cache.shape[5]), 
                                        dtype=kv_cache.dtype, device=kv_cache.device)
    
    return conditioning_tensor