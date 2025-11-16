import torch 
import time
import pickle
import sys
import os
from ssd.engine.sequence import Sequence
from ssd.utils.context import set_context, get_context, reset_context

EXIT_ON_MISMATCH = True
SAVE_DEBUG_DATA = True

def _compare_kv_caches_debug(config, tokenizer, seqs: list[Sequence], target_runner, draft_runner, verbose: bool = True, verify: bool = False) -> bool:
    """
    Compare KV cache states between target and draft runners for sequence blocks.

    Returns:
        bool: True if any mismatches found, False if all caches match
    """
    # Compute KV cache hashes before comparison to ensure read-only operation
    target_hash_before, draft_hash_before = _compute_kv_cache_hash(
        seqs, target_runner, draft_runner)

    if verbose:
        print(f"[Debug] KV Cache comparison for sequence blocks:", flush=True)

    mismatch = False
    for seq_idx, seq in enumerate(seqs):
        if verbose:
            print(
                f"  Seq {seq_idx}: num_cached_tokens={seq.num_cached_tokens}, num_draft_cached_tokens={seq.num_draft_cached_tokens}", flush=True)

        # Get block IDs for target and draft
        num_target_blocks = (
            seq.num_cached_tokens + config.kvcache_block_size - 1) // config.kvcache_block_size
        num_draft_blocks = (seq.num_draft_cached_tokens +
                            config.kvcache_block_size - 1) // config.kvcache_block_size

        target_blocks = seq.block_table[:num_target_blocks]
        draft_blocks = seq.draft_block_table[:num_draft_blocks]

        if verbose:
            print(
                f"  Seq {seq_idx}: target_blocks={target_blocks}, draft_blocks={draft_blocks}", flush=True)

        # Compare KV cache content for overlapping blocks across all layers
        min_blocks = min(len(target_blocks), len(draft_blocks))
        print(f'in compare_kv_cache using min_blocks={min_blocks}')
        if verbose:
            print(
                f'about to check, block_sz={config.kvcache_block_size}')

        for block_idx in range(min_blocks):
            target_block_id = target_blocks[block_idx]
            draft_block_id = draft_blocks[block_idx]

            # Check only first two and last two layers
            num_layers = target_runner.kv_cache.shape[1]
            layers_to_check = []

            # Add first two layers
            if num_layers >= 1:
                layers_to_check.append(0)
            if num_layers >= 2:
                layers_to_check.append(1)

            # Add last two layers (if different from first two)
            if num_layers >= 3:
                layers_to_check.append(num_layers - 2)
            if num_layers >= 4:
                layers_to_check.append(num_layers - 1)
            elif num_layers == 3:
                layers_to_check.append(num_layers - 1)

            # Remove duplicates while preserving order
            layers_to_check = list(dict.fromkeys(layers_to_check))

            block_has_mismatch = False
            layer_stats = []
            D = target_runner.kv_cache.shape[-1] * \
                target_runner.kv_cache.shape[-2]

            for layer_idx in layers_to_check:
                # Get KV cache for this block from current layer
                # kv_cache shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
                # [block_size, num_kv_heads, head_dim]
                target_k_cache = target_runner.kv_cache[0,
                                                        layer_idx, target_block_id, :, :, :]
                target_v_cache = target_runner.kv_cache[1,
                                                        layer_idx, target_block_id, :, :, :]
                draft_k_cache = draft_runner.kv_cache[0,
                                                        layer_idx, draft_block_id, :, :, :]
                draft_v_cache = draft_runner.kv_cache[1,
                                                        layer_idx, draft_block_id, :, :, :]

                # For the last block (if it's partially filled), only compare the valid tokens
                is_last_target_block = (
                    block_idx == len(target_blocks) - 1)
                is_last_draft_block = (block_idx == len(draft_blocks) - 1)

                if is_last_target_block or is_last_draft_block:
                    target_last_block_num_tokens = seq.num_cached_tokens % config.kvcache_block_size
                    draft_last_block_num_tokens = seq.num_draft_cached_tokens % config.kvcache_block_size

                    # Determine valid tokens based on which blocks are last
                    if is_last_target_block and is_last_draft_block:
                        # Both are last blocks, use minimum
                        valid_tokens = min(
                            target_last_block_num_tokens, draft_last_block_num_tokens)
                    elif is_last_target_block:
                        # Only target is last block, use target's token count
                        valid_tokens = target_last_block_num_tokens
                    else:
                        # Only draft is last block, use draft's token count
                        valid_tokens = draft_last_block_num_tokens

                    print(
                        f'in compare_kv_cache using valid_tokens for last block={valid_tokens}')

                    # Compare only the valid portion of the caches
                    target_k_cache_valid = target_k_cache[:valid_tokens, :, :]
                    target_v_cache_valid = target_v_cache[:valid_tokens, :, :]
                    draft_k_cache_valid = draft_k_cache[:valid_tokens, :, :]
                    draft_v_cache_valid = draft_v_cache[:valid_tokens, :, :]

                    k_equal = torch.equal(
                        target_k_cache_valid, draft_k_cache_valid)
                    v_equal = torch.equal(
                        target_v_cache_valid, draft_v_cache_valid)

                    if not k_equal or not v_equal:
                        k_diff = torch.max(torch.abs(
                            target_k_cache_valid - draft_k_cache_valid)).item() if not k_equal else 0.0
                        v_diff = torch.max(torch.abs(
                            target_v_cache_valid - draft_v_cache_valid)).item() if not v_equal else 0.0
                        # Count mismatches for partial blocks
                        if not k_equal:
                            # Check which token positions differ (any difference across head and dim dimensions)
                            k_pos_diffs = torch.any(~torch.isclose(
                                target_k_cache_valid, draft_k_cache_valid, atol=1e-5), dim=(1, 2))
                            k_mismatches = torch.sum(k_pos_diffs).item()
                            k_total = valid_tokens
                            # Get indices of differing positions
                            k_diff_indices = torch.where(
                                k_pos_diffs)[0].tolist()
                        else:
                            k_mismatches = 0
                            k_total = valid_tokens
                            k_diff_indices = []

                        if not v_equal:
                            # Check which token positions differ (any difference across head and dim dimensions)
                            v_pos_diffs = torch.any(~torch.isclose(
                                target_v_cache_valid, draft_v_cache_valid, atol=1e-5), dim=(1, 2))
                            v_mismatches = torch.sum(v_pos_diffs).item()
                            v_total = valid_tokens
                            # Get indices of differing positions
                            v_diff_indices = torch.where(
                                v_pos_diffs)[0].tolist()
                        else:
                            v_mismatches = 0
                            v_total = valid_tokens
                            v_diff_indices = []

                        layer_stats.append({
                            'layer_idx': layer_idx,
                            'block_type': 'partial',
                            'k_equal': k_equal,
                            'v_equal': v_equal,
                            'k_diff': k_diff,
                            'v_diff': v_diff,
                            'k_mismatches': k_mismatches,
                            'k_total': k_total,
                            'v_mismatches': v_mismatches,
                            'v_total': v_total,
                            'k_diff_indices': k_diff_indices,
                            'v_diff_indices': v_diff_indices
                        })
                        block_has_mismatch = True
                    else:
                        layer_stats.append({
                            'layer_idx': layer_idx,
                            'block_type': 'partial',
                            'k_equal': True,
                            'v_equal': True,
                            'k_diff': 0.0,
                            'v_diff': 0.0,
                            'k_mismatches': 0,
                            'k_total': valid_tokens,
                            'v_mismatches': 0,
                            'v_total': valid_tokens,
                            'k_diff_indices': [],
                            'v_diff_indices': []
                        })

                else:
                    # Full block comparison
                    k_equal = torch.equal(target_k_cache, draft_k_cache)
                    v_equal = torch.equal(target_v_cache, draft_v_cache)

                    if not k_equal or not v_equal:
                        k_diff = torch.max(
                            torch.abs(target_k_cache - draft_k_cache)).item() if not k_equal else 0.0
                        v_diff = torch.max(
                            torch.abs(target_v_cache - draft_v_cache)).item() if not v_equal else 0.0
                        # Count mismatches for full blocks - aggregate by position across heads/dims
                        if not k_equal:
                            # Check which token positions differ (any difference across head and dim dimensions)
                            k_pos_diffs = torch.any(~torch.isclose(
                                target_k_cache, draft_k_cache, atol=1e-5), dim=(1, 2))
                            k_mismatches = torch.sum(k_pos_diffs).item()
                            k_total = config.kvcache_block_size
                            # Get indices of differing positions
                            k_diff_indices = torch.where(
                                k_pos_diffs)[0].tolist()
                        else:
                            k_mismatches = 0
                            k_total = config.kvcache_block_size
                            k_diff_indices = []

                        if not v_equal:
                            # Check which token positions differ (any difference across head and dim dimensions)
                            v_pos_diffs = torch.any(~torch.isclose(
                                target_v_cache, draft_v_cache, atol=1e-5), dim=(1, 2))
                            v_mismatches = torch.sum(v_pos_diffs).item()
                            v_total = config.kvcache_block_size
                            # Get indices of differing positions
                            v_diff_indices = torch.where(
                                v_pos_diffs)[0].tolist()
                        else:
                            v_mismatches = 0
                            v_total = config.kvcache_block_size
                            v_diff_indices = []

                        layer_stats.append({
                            'layer_idx': layer_idx,
                            'block_type': 'full',
                            'k_equal': k_equal,
                            'v_equal': v_equal,
                            'k_diff': k_diff,
                            'v_diff': v_diff,
                            'k_mismatches': k_mismatches,
                            'k_total': k_total,
                            'v_mismatches': v_mismatches,
                            'v_total': v_total,
                            'k_diff_indices': k_diff_indices,
                            'v_diff_indices': v_diff_indices
                        })
                        block_has_mismatch = True
                    else:
                        layer_stats.append({
                            'layer_idx': layer_idx,
                            'block_type': 'full',
                            'k_equal': True,
                            'v_equal': True,
                            'k_diff': 0.0,
                            'v_diff': 0.0,
                            'k_mismatches': 0,
                            'k_total': config.kvcache_block_size,
                            'v_mismatches': 0,
                            'v_total': config.kvcache_block_size,
                            'k_diff_indices': [],
                            'v_diff_indices': []
                        })

            # Print stats for all layers if any mismatch was found in this block
            if block_has_mismatch and verbose:
                print(
                    f"    MISMATCH FOUND - Block {block_idx}: target_id={target_block_id}, draft_id={draft_block_id}", flush=True)
                for stat in layer_stats:
                    k_indices_str = f" at indices {stat['k_diff_indices']}" if stat['k_diff_indices'] else ""
                    v_indices_str = f" at indices {stat['v_diff_indices']}" if stat['v_diff_indices'] else ""
                    print(f"      Layer {stat['layer_idx']} ({stat['block_type']}): K {stat['k_mismatches']}/{stat['k_total']} pos differ (max_diff={stat['k_diff']:.6f}){k_indices_str}, V {stat['v_mismatches']}/{stat['v_total']} pos differ (max_diff={stat['v_diff']:.6f}){v_indices_str}", flush=True)

                # Compute and print L1/L2 differences for this block
                for layer_idx in layers_to_check:
                    target_k_cache = target_runner.kv_cache[0,
                                                            layer_idx, target_block_id]
                    target_v_cache = target_runner.kv_cache[1,
                                                            layer_idx, target_block_id]
                    draft_k_cache = draft_runner.kv_cache[0,
                                                            layer_idx, draft_block_id]
                    draft_v_cache = draft_runner.kv_cache[1,
                                                            layer_idx, draft_block_id]

                    k_l1_diff = torch.sum(
                        torch.abs(target_k_cache - draft_k_cache)).item()
                    k_l2_diff = torch.sqrt(
                        torch.sum((target_k_cache - draft_k_cache) ** 2)).item()
                    v_l1_diff = torch.sum(
                        torch.abs(target_v_cache - draft_v_cache)).item()
                    v_l2_diff = torch.sqrt(
                        torch.sum((target_v_cache - draft_v_cache) ** 2)).item()

                    print(
                        f"      Layer {layer_idx} L1/L2 diffs: K L1={k_l1_diff:.6f} L2={k_l2_diff:.6f}, V L1={v_l1_diff:.6f} L2={v_l2_diff:.6f}", flush=True)

                mismatch = True

    # Compute KV cache hashes after comparison to ensure read-only operation
    target_hash_after, draft_hash_after = _compute_kv_cache_hash(
        seqs, target_runner, draft_runner)

    # Verify that the comparison didn't modify the caches
    if target_hash_before != target_hash_after or draft_hash_before != draft_hash_after:
        print(
            f"[_compare_kv_caches] ERROR: KV cache comparison modified the caches!")
        print(
            f"[_compare_kv_caches] Target hash: {target_hash_before} -> {target_hash_after}")
        print(
            f"[_compare_kv_caches] Draft hash: {draft_hash_before} -> {draft_hash_after}")
        
        if EXIT_ON_MISMATCH:
            sys.exit()

    if verbose:
        if mismatch:
            print(f"[Debug] KV Cache comparison: MISMATCHES FOUND", flush=True)
        else:
            print(f"[Debug] KV Cache comparison: no mismatches :D", flush=True)

    # If mismatch found, print debug information
    if mismatch:
        # print the sequence state and use that as prompt in our test
        print(f"[_compare_kv_caches] [KV Cache Mismatch Debug] seqs:")
        for i, seq in enumerate(seqs):
            detokenized_token_ids = tokenizer.decode(seq.token_ids)
            print(
                f"[_compare_kv_caches]   Seq {i} (id={seq.seq_id}): token_ids={seq.token_ids} -> \"{detokenized_token_ids}\"")
        print(f'[_compare_kv_caches] [KV Cache Mismatch Debug] ---')
        if verify:
            if EXIT_ON_MISMATCH: sys.exit()

    return mismatch

def _compute_kv_cache_hash(seqs: list[Sequence], target_runner, draft_runner) -> tuple[int, int]:
    """
    Compute hash of KV cache states for both runners to detect changes.
    Only hashes the blocks that are actually used by the sequences.

    Returns:
        tuple[int, int]: (target_hash, draft_hash)
    """
    import hashlib

    from copy import deepcopy
    # Collect all block IDs that are in use
    target_block_ids = set()
    draft_block_ids = set()

    seqs = deepcopy(seqs)
    for seq in seqs:
        target_block_ids.update(seq.block_table[:seq.num_cached_blocks])
        draft_block_ids.update(
            seq.draft_block_table[:seq.num_draft_cached_blocks])

    # Hash target KV cache for used blocks only
    target_hasher = hashlib.sha256()
    for block_id in sorted(target_block_ids):
        # Hash a small sample from each layer (first and last layer, first few positions)
        for layer_idx in [0, target_runner.kv_cache.shape[1] - 1]:
            # Sample first 4 positions of the block, convert to float32 to avoid BFloat16 issue
            # Use .detach() to ensure we don't modify gradients or computation graph
            sample_k = target_runner.kv_cache[0, layer_idx, block_id, :4, 0, :8].detach(
            ).float().cpu().numpy().tobytes()
            sample_v = target_runner.kv_cache[1, layer_idx, block_id, :4, 0, :8].detach(
            ).float().cpu().numpy().tobytes()
            target_hasher.update(sample_k)
            target_hasher.update(sample_v)

    # Hash draft KV cache for used blocks only
    draft_hasher = hashlib.sha256()
    for block_id in sorted(draft_block_ids):
        # Hash a small sample from each layer (first and last layer, first few positions)
        for layer_idx in [0, draft_runner.kv_cache.shape[1] - 1]:
            # Sample first 4 positions of the block, convert to float32 to avoid BFloat16 issue
            # Use .detach() to ensure we don't modify gradients or computation graph
            sample_k = draft_runner.kv_cache[0, layer_idx, block_id, :4, 0, :8].detach(
            ).float().cpu().numpy().tobytes()
            sample_v = draft_runner.kv_cache[1, layer_idx, block_id, :4, 0, :8].detach(
            ).float().cpu().numpy().tobytes()
            draft_hasher.update(sample_k)
            draft_hasher.update(sample_v)

    target_hash = int(target_hasher.hexdigest()[:16], 16)
    draft_hash = int(draft_hasher.hexdigest()[:16], 16)

    return target_hash, draft_hash

def sleep_for_small_target_debug(config):
    if not __debug__: return 
    
    if ("0.6B" in config.model or "1B" in config.model or "4B" in config.model or "8B" in config.model):
        sleep_time = 0.01
        if config.enforce_eager or __debug__:
            sleep_time = 0.25
        time.sleep(sleep_time)


def _compare_logits_and_save_state_debug(config, target_runner, logits_p, logits_p_decode, batch_size, seqs_copy) -> bool:
    # Compare logits_p (from verify) with logits_p_decode (from speculate_sync) for debugging
    if logits_p_decode is not None:
        # Both should be [B, K+1, V]
        assert logits_p.shape == logits_p_decode.shape, f"Shape mismatch: logits_p {logits_p.shape} vs logits_p_decode {logits_p_decode.shape}"

        # Check for equality with some tolerance for floating point precision
        tolerance = 1e-5
        logits_match = torch.allclose(
            logits_p, logits_p_decode, atol=tolerance, rtol=tolerance)

        if not logits_match:
            # Find positions where they disagree along the second axis (K+1 positions)
            # A position disagrees if any element at that position disagrees
            diff_mask = ~torch.isclose(
                logits_p, logits_p_decode, atol=tolerance, rtol=tolerance)
            # Check disagreement per position across all sequences and vocab
            # [K+1] - True if any seq/vocab disagrees at this position
            position_disagreements = diff_mask.any(dim=(0, 2))
            num_position_disagreements = position_disagreements.sum().item()
            max_diff = torch.abs(logits_p - logits_p_decode).max().item()

            # Find which specific positions disagree
            disagreeing_positions = torch.where(position_disagreements)[
                0].cpu().tolist()

            print(
                f"[verify] LOGITS MISMATCH: logits_p vs logits_p_decode disagree at {num_position_disagreements} positions (out of {logits_p.shape[1]})", flush=True)
            print(
                f"[verify] Disagreeing positions: {disagreeing_positions}", flush=True)
            print(
                f"[verify] Maximum difference: {max_diff:.6f}", flush=True)

            # Print per-sequence disagreement info
            for b in range(batch_size):
                seq_diff_mask = diff_mask[b]  # [K+1, V]
                # [K+1] - True if any vocab disagrees at this position
                seq_position_disagreements = seq_diff_mask.any(dim=1)
                seq_num_position_disagreements = seq_position_disagreements.sum().item()
                if seq_num_position_disagreements > 0:
                    seq_disagreeing_positions = torch.where(
                        seq_position_disagreements)[0].cpu().tolist()
                    seq_max_diff = torch.abs(
                        logits_p[b] - logits_p_decode[b]).max().item()
                    print(
                        f"[verify] Seq {b}: {seq_num_position_disagreements} position disagreements at positions {seq_disagreeing_positions}, max diff: {seq_max_diff:.6f}", flush=True)
                    
                    # Print first two entries of verify logits at each sequence position
                    for pos in range(logits_p.shape[1]):  # K+1 positions
                        verify_logits_first_two = logits_p[b, pos, :2].tolist()
                        decode_logits_first_two = logits_p_decode[b, pos, :2].tolist()
                        print(
                            f"[verify] Seq {b} Pos {pos}: verify_logits[:2]={verify_logits_first_two}, decode_logits[:2]={decode_logits_first_two}", flush=True)

            if SAVE_DEBUG_DATA:
                # Pickle debug data for analysis
                debug_dir = "/data/tkumar/debug"
                os.makedirs(debug_dir, exist_ok=True)
                print(f'DEBUG SAVING DATA')

                # Save KV caches from both models
                target_kv_cache = target_runner.kv_cache

                # Print detokenized sequence for debugging
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config.model)
                detokenized_seq = tokenizer.decode(seqs_copy[0].token_ids, skip_special_tokens=True)
                print(f"Detokenized sequence causing mismatch: '{detokenized_seq}'")
                
                # Save sequences and config
                debug_data = {
                    'target_kv_cache': target_kv_cache.cpu(),
                    'seqs_copy': seqs_copy,
                    'target_config': config,
                }

                debug_file = os.path.join(
                    debug_dir, "logits_mismatch_debug.pkl")
                with open(debug_file, 'wb') as f:
                    pickle.dump(debug_data, f)

                print(
                    f"[verify] Debug data saved to {debug_file}", flush=True)
            if EXIT_ON_MISMATCH:
                sys.exit()
        else:
            print(
                f"[verify] logits_p and logits_p_decode match within tolerance {tolerance}", flush=True)

def make_dummy_seq(tokenizer):
    from ssd.engine.sequence import Sequence
    from ssd.sampling_params import SamplingParams
    
    # Create a dummy prompt
    dummy_prompt = "Hello, how are you today?"
    token_ids = tokenizer.encode(dummy_prompt)
    
    # Create sequence with dummy prompt
    seq = Sequence(token_ids, SamplingParams())
    
    # Set block tables to use block 10000
    seq.block_table = [10000]
    seq.draft_block_table = [10000]
    
    # Set cached tokens to 0 (will prefill)
    seq.num_cached_tokens = 0
    seq.num_draft_cached_tokens = 0
    
    # Set num_tokens to length of tokenized prompt
    seq.num_tokens = len(token_ids)
    
    return seq


def sample_decode_steps_after_prefill(model_runner, initial_token_ids, cu_q, block_tables, batch_size, num_steps):
    """Stateless helper to sample decode steps."""
    current_tokens = torch.tensor(initial_token_ids, dtype=torch.int64, device=model_runner.device)
    all_generated_tokens = [initial_token_ids]
    
    for step in range(num_steps):
        # Calculate positions for each sequence based on cu_q lengths
        positions = torch.zeros(batch_size, dtype=torch.int64, device=model_runner.device)
        decode_slot_mapping = torch.zeros(batch_size, dtype=torch.int32, device=model_runner.device)
        
        for i in range(batch_size):
            seq_len = cu_q[i+1] - cu_q[i]  # length of sequence i from cu_q
            positions[i] = seq_len + step  # next position after prefill + decode steps so far
            
            # Use the slot contiguously after the prefill slots for this sequence
            prefill_end_slot = cu_q[i+1] - 1
            decode_slot_mapping[i] = prefill_end_slot + step  # next slot after prefill slots

        fdbt = block_tables.clone()
        if fdbt[0, 1] == -1: 
            fdbt[0, 1] = fdbt[0, 0] + 1 
            # print(f'hacky faking another block for debug decode') # "ia Natal Natal Natal Natal and" w/o zero and  -- so our prefill context isn't working right 
        
        # Set up decode context
        context_lens = (positions.clone() + 1).to(torch.int32)  # current length of each sequence, ensure int32 dtype
        set_context(is_prefill=False, cu_seqlens_q=None, cu_seqlens_k=None, 
                max_seqlen_q=None, max_seqlen_k=-1, slot_mapping=decode_slot_mapping, 
                context_lens=context_lens, block_tables=fdbt)
        # Run decode step
        # print(f'[debug decode step {step}] current_tokens.shape={current_tokens.shape}, positions.shape={positions.shape}', flush=True)
        decode_logits = model_runner.run_model(current_tokens, positions, is_prefill=False, last_only=True)
        # print(f'[debug decode step {step}] decode_logits.shape={decode_logits.shape}', flush=True)
        
        # Sample next tokens - fix temperature tensor shape
        temperatures = torch.zeros(decode_logits.shape[0], device=model_runner.device)
        # print(f'[debug decode step {step}] temperatures.shape={temperatures.shape}', flush=True)
        next_token_ids = model_runner.sampler(decode_logits, temperatures).tolist() 
        # print(f'[debug decode step {step}] next_token_ids len={len(next_token_ids)}, values={next_token_ids}', flush=True)
        current_tokens = torch.tensor(next_token_ids, dtype=torch.int64, device=model_runner.device)
        # print(f'[debug decode step {step}] updated current_tokens.shape={current_tokens.shape}', flush=True)
        all_generated_tokens.append(next_token_ids)
        # print(f'[debug decode step {step}] all_generated_tokens len={len(all_generated_tokens)}', flush=True)
        
        # Clean up context
        reset_context()
    
    return all_generated_tokens


    def sanity_check_kvcache_nonempty(kv_cache, dbt, positions, block_size, q_len=None, is_prefill=False):
        # check all positions we condition on do not have any pos/cols that are allzero, and that everything in slot_map positions is zero
        B = dbt.shape[0]

        if is_prefill:
            # For prefill, just check all positions supplied without any q_len adjustments
            unique_positions = torch.unique(positions)
            for pos in unique_positions:
                pos_val = pos.item()
                block_idx = pos_val // block_size
                offset = pos_val % block_size

                # print(f'checking position {pos_val} in kv cache (prefill mode)', flush=True)
                # Get the actual block ID from dbt at this block index
                # assuming single batch for prefill
                actual_block_id = dbt[0, block_idx].item()

                # kv_cache shape: [2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim]

                # Check that this position in KV cache is not all zero across ALL layers
                # [2, layers, num_heads, head_dim]
                kv_cache_slice = kv_cache[:, :, actual_block_id, offset, :, :]

                kv_allzeros = torch.allclose(
                    kv_cache_slice, torch.zeros_like(kv_cache_slice))  # should not be true

                if kv_allzeros:
                    print(
                        f"ASSERT FAILED -- Batch {b}: Both K and V cache at position {pos} (block_idx {block_idx}, actual_block_id {actual_block_id}, offset {offset}) are ONES, but this position should have been filled")
                    exit()
        else:
            # Original decode logic
            assert q_len is not None, "ERROR in sanity_check_kvcache_nonempty: q_len must be provided for decode"
            for b in range(B):
                # Get positions for this batch element
                batch_positions = positions[b * q_len:(b + 1) * q_len]

                # Only check up to positions.max() - q_len since we'll store new tokens in forward pass
                # this assumes positions vector is contiguous? should be...
                max_pos = batch_positions.max().item()
                check_up_to_pos = max_pos - q_len
                assert check_up_to_pos == batch_positions.min().item() - 1

                # If check_up_to_pos < 0, there's nothing to check for this batch
                if check_up_to_pos < 0:
                    continue

                # Check that used blocks don't have any positions/columns that are all zero
                # We check all positions from 0 to check_up_to_pos for any blocks that are referenced
                for pos in range(check_up_to_pos + 1):
                    block_idx = pos // block_size
                    offset = pos % block_size

                    # Get the actual block ID from dbt for this batch element
                    actual_block_id = dbt[b, block_idx].item()

                    # print(f'checking position {pos} in kv cache', flush=True)
                    # kv_cache shape: [2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim]

                    # Check that this position in KV cache is not all zero across ALL layers
                    # [2, layers, num_heads, head_dim]
                    kv_cache_slice = kv_cache[:, :,
                                              actual_block_id, offset, :, :]

                    kv_allzeros = torch.allclose(
                        kv_cache_slice, torch.zeros_like(kv_cache_slice))  # should not be true, ie. we want it to have deviated from init after prefill/before running

                    if kv_allzeros:
                        print(
                            f"ASSERT FAILED -- Batch {b}: Both K and V cache at position {pos} (block_idx {block_idx}, actual_block_id {actual_block_id}, offset {offset}) are ONES, but this position should have been filled")
                        exit()
