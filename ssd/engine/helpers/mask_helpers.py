import torch 

# Global cache for precomputed mask components
_mask_cache = {
    'glue_and_rec_mask': None,
    'diag_components': None,
    'ones_tensor': None,
    'cached_params': None
}

@torch.inference_mode()
def get_mask_iter_i(i: int, prefix_len: int, K: int, F: int) -> torch.Tensor:
    q_len = F * (K + 1)
    prefix_mask = torch.ones(q_len, prefix_len)  # prefix_len = nt-1
    glue_and_rec_mask = (torch.arange(K+1) <= torch.arange(K+1).unsqueeze(1) 
                         ).repeat_interleave(F, dim=0).to(torch.int32)
    diags = [torch.diag(torch.ones(q_len)) for _ in range(i+1)] 
    mask = torch.cat([prefix_mask, glue_and_rec_mask, *diags], dim=1)
    assert mask.size(0) == q_len, f"ERROR in get_mask_iter_i: mask should have length q_len, got {mask.size(0)}"
    assert mask.size(1) == prefix_len + (K+1) + (i+1) * q_len, f"ERROR in get_mask_iter_i: mask should have length q_len + (K+1) + (i+1) * q_len, got {mask.size(1)}"
    return mask.to(torch.bool)

# TODO: now glue_and_rec becomes [b] depedent based on cache_hits
@torch.inference_mode()
def _precompute_mask_components(K: int, F: int, max_step: int, max_context_len: int, device: torch.device, fan_out_list: torch.Tensor, fan_out_list_miss: torch.Tensor):
    """Precompute the static components of masks including a large ones tensor for prefix masks."""
    # Precompute glue_and_rec_mask for both cache hit and miss cases
    new_row_idx_hit = torch.arange(K+1, device=device).repeat_interleave(fan_out_list)
    glue_and_rec_mask_hit = torch.tril(torch.ones(K+1, K+1, device=device), diagonal=0)[new_row_idx_hit].to(torch.int32)
    
    new_row_idx_miss = torch.arange(K+1, device=device).repeat_interleave(fan_out_list_miss)
    glue_and_rec_mask_miss = torch.tril(torch.ones(K+1, K+1, device=device), diagonal=0)[new_row_idx_miss].to(torch.int32)
    
    MQ_LEN = glue_and_rec_mask_hit.shape[0]
    assert glue_and_rec_mask_miss.shape[0] == MQ_LEN, f"MQ_LEN mismatch: hit={MQ_LEN}, miss={glue_and_rec_mask_miss.shape[0]}"

    # Precompute diagonal portions for each step
    diag_components = {}
    for step in range(max_step + 1):
        diags = [torch.diag(torch.ones(MQ_LEN, device=device)) for _ in range(step + 1)]
        if diags:
            diag_components[step] = torch.cat(diags, dim=1)
        else:
            diag_components[step] = torch.empty(MQ_LEN, 0, device=device)
    
    # Precompute a large ones tensor for prefix masks
    ones_tensor = torch.ones(MQ_LEN, max_context_len, device=device)
    
    return glue_and_rec_mask_hit, glue_and_rec_mask_miss, diag_components, ones_tensor


def _get_custom_mask_optimized(context_lens, step: int, K: int, F: int, B: int, device: torch.device, 
                              glue_and_rec_mask_hit, glue_and_rec_mask_miss, diag_components, ones_tensor, cache_hits):
    """Optimized mask computation using precomputed components and slicing from large ones tensor."""
    MQ_LEN = glue_and_rec_mask_hit.shape[0]
    glue_added = K + 1
    tree_decode_added = (step + 1) * MQ_LEN
    ttl_added = tree_decode_added + glue_added

    masks = []
    for b in range(B):
        prefix_len = context_lens[b] - ttl_added
        assert prefix_len >= 0, f"ERROR in get_custom_mask: prefix_len should be non-negative, got {prefix_len}"
        
        # Slice from precomputed ones tensor instead of creating new tensor
        prefix_mask = ones_tensor[:, :prefix_len]
        
        # Select appropriate glue_and_rec_mask based on cache hit status
        if cache_hits[b] == 1:
            glue_and_rec_mask = glue_and_rec_mask_hit
        else:
            glue_and_rec_mask = glue_and_rec_mask_miss
        
        # Use precomputed components
        mask = torch.cat([prefix_mask, glue_and_rec_mask, diag_components[step]], dim=1)
        
        assert mask.shape == (MQ_LEN, context_lens[b]), f"ERROR in get_custom_mask: mask should have shape {(MQ_LEN, context_lens[b])}, got {mask.shape}"
        masks.append(mask.view(-1))

    return torch.cat(masks, dim=0).to(torch.bool)


@torch.inference_mode()
def get_custom_mask_cached(config, context_lens, step: int, K: int, F: int, B: int, device: torch.device, fan_out_list: list[int], fan_out_list_miss: list[int], cache_hits: torch.Tensor):
    global _mask_cache
    
    # Max step will be K+1
    max_step = K + 1
    
    # Convert fan_out_lists to tensors for consistent caching
    fan_out_tensor = torch.tensor(fan_out_list, dtype=torch.int64, device=device)
    fan_out_tensor_miss = torch.tensor(fan_out_list_miss, dtype=torch.int64, device=device)
    
    # Current parameters for cache validation (exclude cache_hits since it varies per call)
    current_params = (K, F, max_step, config.max_model_len, device, tuple(fan_out_list), tuple(fan_out_list_miss))
    
    # Check if we need to precompute or recompute
    if (_mask_cache['cached_params'] is None or 
        _mask_cache['cached_params'] != current_params):
        
        # Precompute mask components
        glue_and_rec_mask_hit, glue_and_rec_mask_miss, diag_components, ones_tensor = _precompute_mask_components(
            K, F, max_step, config.max_model_len, device, fan_out_tensor, fan_out_tensor_miss)
        
        # Update cache
        _mask_cache['glue_and_rec_mask_hit'] = glue_and_rec_mask_hit
        _mask_cache['glue_and_rec_mask_miss'] = glue_and_rec_mask_miss
        _mask_cache['diag_components'] = diag_components
        _mask_cache['ones_tensor'] = ones_tensor
        _mask_cache['cached_params'] = current_params
    
    # Use optimized computation with cached components
    mask = _get_custom_mask_optimized(
        context_lens, step, K, F, B, device,
        _mask_cache['glue_and_rec_mask_hit'],
        _mask_cache['glue_and_rec_mask_miss'],
        _mask_cache['diag_components'],
        _mask_cache['ones_tensor'],
        cache_hits,
    )

    return mask


@torch.inference_mode()
def get_custom_mask_vectorized(context_lens, step: int, K: int, F: int, B: int, device: torch.device):
    """Vectorized version of get_custom_mask using ragged concatenation approach."""
    MQ_LEN = F * (K+1)
    q_len = MQ_LEN
    glue_added = K + 1
    tree_decode_added = (step + 1) * MQ_LEN
    ttl_added = tree_decode_added + glue_added

    # Calculate prefix lengths for each batch element
    prefix_lens = context_lens - ttl_added
    assert torch.all(
        prefix_lens >= 0), f"ERROR: prefix_lens should be non-negative, got {prefix_lens}"

    # Build glue_and_rec_mask: [q_len, K+1]
    glue_and_rec_mask = (torch.arange(K+1, device=device) <= torch.arange(K+1, device=device).unsqueeze(1)
                         ).repeat_interleave(F, dim=0).to(torch.int32)

    # Build diagonal matrices: [q_len, (step+1) * q_len]
    diag_matrices = []
    for i in range(step + 1):
        diag_matrices.append(torch.diag(torch.ones(q_len, device=device)))
    diag_combined = torch.cat(
        diag_matrices, dim=1) if diag_matrices else torch.zeros(q_len, 0, device=device)

    # Combine glue_and_rec_mask with diagonal matrices to form M
    M = torch.cat([glue_and_rec_mask, diag_combined], dim=1).to(torch.bool)

    # Use the flat_blocks_after_cat pattern
    return flat_blocks_after_cat(prefix_lens, M)


@torch.inference_mode()
def flat_blocks_after_cat(L: torch.Tensor, M: torch.Tensor):
    """
    L: 1D long/int tensor [k] with positive lengths
    M: [N, y] tensor (constant block appended to each [N, L[i]] along dim=1)
    Returns:
      1D tensor equal to cat_i( flatten( [ones(N, L[i]) cat M] ) ).
    """
    assert L.ndim == 1 and L.numel() > 0
    N, y = M.shape
    k = L.numel()
    device, dtype = M.device, M.dtype

    cols_per_block = L.to(torch.long) + y                 # [k]
    total_cols = int(cols_per_block.sum().item())

    # one allocation (all ones for the variable-length left parts)
    T = torch.ones((N, total_cols), device=device, dtype=dtype)

    # mark the last y columns of each block (vectorized) and write M everywhere at once
    offs = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                      cols_per_block.cumsum(0)[:-1]])     # [k] start col per block
    pos = torch.arange(total_cols, device=device)         # [total_cols]
    blk = torch.repeat_interleave(torch.arange(
        k, device=device), cols_per_block)  # [total_cols]
    # 0..(L[i]+y-1) per block
    within = pos - offs[blk]
    # last y columns per block
    mask = within.ge(L[blk])
    # single strided write
    T[:, mask] = M.repeat(1, k)

    # views per block, then flatten each view and cat once
    # list of [N, L[i]+y] VIEWS
    blocks = T.split(cols_per_block.tolist(), dim=1)
    out = torch.cat([b.reshape(-1) for b in blocks], dim=0)
    return out


# we should add support for BS>8 of nonuniform fan out 
# to get more favorable BS scaling (can afford half the number of fan out tokens)
@torch.inference_mode()
def get_custom_mask(config, context_lens, step: int, K: int, F: int, B: int, device: torch.device, cache_hits: torch.Tensor):
    if B <= 8: 
        return get_custom_mask_cached(config, context_lens, step, K, F, B, device, fan_out_list=config.fan_out_list, fan_out_list_miss=config.fan_out_list_miss, cache_hits=cache_hits)
    else:
        assert all(f == F for f in config.fan_out_list), f"ERROR in get_custom_mask: all entries in fan_out_list must be {F} at BS>8"
        assert all(f == F for f in config.fan_out_list_miss), f"ERROR in get_custom_mask: all entries in fan_out_list_miss must be {F} at BS>8"
        return get_custom_mask_vectorized(context_lens, step, K, F, B, device)

