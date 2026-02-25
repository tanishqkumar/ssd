import os
import torch
import numpy as np
from typing import List
from ssd.utils.context import set_context, get_context, reset_context
from ssd.engine.helpers.mask_helpers import get_custom_mask
from flashinfer.quantization import get_quantization_module
from time import perf_counter


# PERFORMANCE: CPU-side mask indptr avoids GPU->CPU sync inside FlashInfer plan().
# Replicates FlashInfer's _compute_page_mask_indptr using only CPU tensors.
def _compute_page_mask_indptr_cpu(qo_indptr, paged_kv_indptr, paged_kv_last_page_len, page_size):
    mask_indptr = torch.empty_like(qo_indptr)
    mask_indptr[0] = 0
    mask_indptr[1:] = torch.cumsum(
        (qo_indptr[1:] - qo_indptr[:-1])
        * ((paged_kv_indptr[1:] - paged_kv_indptr[:-1] - 1) * page_size + paged_kv_last_page_len),
        0,
    )
    return mask_indptr


# PERFORMANCE: segment_packbits with CPU-computed metadata avoids GPU->CPU .item() sync.
# FlashInfer's segment_packbits calls indptr_new[-1].item() on a GPU tensor = sync point.
# We compute indptr_new on CPU first, then transfer to GPU async.
def _segment_packbits_no_sync(x_gpu, mask_indptr_cpu, bitorder="little"):
    device = x_gpu.device
    indptr_cpu = mask_indptr_cpu.to(torch.int32)
    seglen = indptr_cpu[1:] - indptr_cpu[:-1]
    packed_len = (seglen + 7) // 8
    indptr_new_cpu = torch.zeros(len(indptr_cpu), dtype=torch.int32)
    indptr_new_cpu[1:] = torch.cumsum(packed_len, 0)
    output_nnzs = int(indptr_new_cpu[-1].item())
    indptr_gpu = indptr_cpu.to(device, non_blocking=True)
    indptr_new_gpu = indptr_new_cpu.to(device, non_blocking=True)
    y = torch.empty(output_nnzs, dtype=torch.uint8, device=device)
    get_quantization_module().segment_packbits(x_gpu, indptr_gpu, indptr_new_gpu, bitorder, y)
    return y, indptr_new_gpu



## RUN CUDAGRAPHS
@torch.inference_mode()
def run_verify_cudagraph(model_runner, input_ids, positions, last_only, graph_vars):
    context = get_context()
    k_plus_1 = model_runner.config.speculate_k + 1
    orig_bs = input_ids.size(0) // k_plus_1  # orig_bs = N here

    wrapper_bs = next(
        x for x in model_runner.graph_bs_list["verify"] if x >= orig_bs)
    graph = model_runner.graphs["verify"][wrapper_bs]

    for k, v in graph_vars.items():
        if k != "outputs":
            v.zero_()

    # Pad to graph bucket size if needed (fixes B>=6 crash from non-monotonic cu_seqlens_q)
    if wrapper_bs > orig_bs:
        pad_bs = wrapper_bs - orig_bs
        pad_flat = pad_bs * k_plus_1
        dev = input_ids.device

        input_ids = torch.cat([input_ids, torch.zeros(pad_flat, dtype=input_ids.dtype, device=dev)])
        positions = torch.cat([positions, torch.zeros(pad_flat, dtype=positions.dtype, device=dev)])
        slot_mapping = torch.cat([
            context.slot_mapping,
            torch.full((pad_flat,), -1, dtype=context.slot_mapping.dtype, device=dev)])
        # Repeat last real row for ghost sequences (valid page table / context len)
        bt = context.block_tables
        cl = context.context_lens
        block_tables = torch.cat([bt, bt[orig_bs-1:orig_bs].expand(pad_bs, -1).contiguous()])
        context_lens = torch.cat([cl, cl[orig_bs-1:orig_bs].expand(pad_bs).contiguous()])
        bs = wrapper_bs
    else:
        slot_mapping = context.slot_mapping
        block_tables = context.block_tables
        context_lens = context.context_lens
        bs = orig_bs

    graph_vars["input_ids"][:bs * k_plus_1] = input_ids
    graph_vars["positions"][:bs * k_plus_1] = positions
    graph_vars["slot_mapping"][:bs * k_plus_1] = slot_mapping
    graph_vars["context_lens"][:bs] = context_lens
    # Construct cu_seqlens_q for FULL padded batch (monotonically increasing)
    seqlen_q = torch.full(
        (bs,), k_plus_1, dtype=torch.int32, device=graph_vars["cu_seqlens_q"].device)
    cu = graph_vars["cu_seqlens_q"][:bs + 1]
    cu.zero_()
    cu[1:].copy_(torch.cumsum(seqlen_q, 0))

    if block_tables is not None:
        graph_vars["block_tables"][:bs, :block_tables.size(1)] = block_tables

    _pt = os.environ.get("SSD_PROFILE_TARGET", "0") == "1"
    if _pt:
        torch.cuda.synchronize()
        _t0 = perf_counter()

    graph.replay()

    if _pt:
        torch.cuda.synchronize()
        _t1 = perf_counter()

    # Extract outputs for the ORIGINAL batch size only
    outputs = graph_vars["outputs"][:orig_bs * k_plus_1]
    logits = model_runner.model.compute_logits(outputs, last_only)

    if _pt:
        torch.cuda.synchronize()
        _t2 = perf_counter()
        has_eagle = "eagle_acts" in graph_vars
        print(f"[cuda_graph_helpers.run_verify_cudagraph][PROFILE verify_cg] replay={(_t1-_t0)*1000:.2f}ms logits={(_t2-_t1)*1000:.2f}ms eagle={has_eagle} bs={orig_bs} rank={model_runner.rank}", flush=True)

    # For eagle target, also return eagle_acts
    if "eagle_acts" in graph_vars:
        eagle_acts = graph_vars["eagle_acts"][:orig_bs * k_plus_1]
        return logits, eagle_acts
    return logits


@torch.inference_mode()
def run_decode_cudagraph(model_runner, input_ids, positions, last_only, graph_vars, hidden_states=None):
    context = get_context()

    flat_batch_size = input_ids.size(0)

    graph = model_runner.graphs["decode"][next(
        x for x in model_runner.graph_bs_list["decode"] if x >= flat_batch_size)]

    for k, v in graph_vars.items():
            if k != "outputs":
                v.zero_()

    graph_vars["input_ids"][:flat_batch_size] = input_ids
    graph_vars["positions"][:flat_batch_size] = positions
    graph_vars["slot_mapping"][:flat_batch_size] = context.slot_mapping
    graph_vars["context_lens"][:flat_batch_size] = context.context_lens

    if hidden_states is not None and "hidden_states" in graph_vars:
        graph_vars["hidden_states"][:flat_batch_size] = hidden_states

    if context.block_tables is not None:
        graph_vars["block_tables"][:flat_batch_size,
                                :context.block_tables.size(1)] = context.block_tables

    graph.replay()

    outputs = graph_vars["outputs"][:flat_batch_size]
    logits = model_runner.model.compute_logits(outputs, last_only)
    # EAGLE draft: outputs is prenorm, return both
    if "hidden_states" in graph_vars:
        return logits, outputs
    return logits


cache = {}

_plan_event = None  # Lazy-init CUDA event for plan() sync
PROFILE = os.environ.get("SSD_PROFILE", "0") == "1"
PROFILE_DRAFT = os.environ.get("SSD_PROFILE_DRAFT", "0") == "1"
_draft_events = []  # [(step, label, start_event, end_event), ...]

def flush_draft_profile():
    """Sync once, read all CUDA events, print per-step breakdown, clear list."""
    if not _draft_events:
        return
    torch.cuda.synchronize()
    by_step = {}
    for step, label, ev0, ev1 in _draft_events:
        by_step.setdefault(step, []).append((label, ev0.elapsed_time(ev1)))
    parts = []
    total = 0.0
    for step in sorted(by_step):
        step_total = sum(t for _, t in by_step[step])
        detail = " ".join(f"{l}={t:.2f}" for l, t in by_step[step])
        parts.append(f"s{step}={step_total:.2f}({detail})")
        total += step_total
    print(f"[cuda_graph_helpers.flush_draft_profile][PROFILE draft_detail] K={len(by_step)} total={total:.2f}ms avg_step={total/len(by_step):.2f}ms | {' '.join(parts)}", flush=True)
    _draft_events.clear()

@torch.inference_mode()
def run_fi_tree_decode_cudagraph(model_runner, input_ids, positions, last_only, graph_vars, step, cache_hits, hidden_states=None):
    # bs != len(input_ids, positions) now in multi-query seting, also need step-dependent mask
    context = get_context()
    assert context.cu_seqlens_q is None, "ERROR in run_fi_tree_decode_cudagraph: cu_seqlens_q should be set to None so we don't take FA path"

    K, F = model_runner.config.speculate_k, model_runner.config.async_fan_out
    # MQ_LEN = F * (K+1)
    MQ_LEN = sum(model_runner.config.fan_out_list)
    orig_flat = input_ids.size(0)
    assert orig_flat % MQ_LEN == 0, f"ERROR in run_fi_tree_decode_cudagraph: flat_batch_size should be divisible by MQ_LEN, got {orig_flat} and {MQ_LEN}"
    orig_B = orig_flat // MQ_LEN

    # Pick CUDA graph and wrapper bucket
    wrapper_bs = next(
        x for x in model_runner.graph_bs_list["fi_tree_decode"] if x >= orig_B)
    graph = model_runner.graphs["fi_tree_decode"][wrapper_bs]
    wrapper = model_runner.prefill_wrappers[wrapper_bs]

    # Prepare padded inputs/context if needed
    if wrapper_bs > orig_B:
        # print(f'PADDING--')
        pad_B = wrapper_bs - orig_B
        pad_flat = pad_B * MQ_LEN

        # Pad queries (ids/rope positions)
        pad_ids = torch.zeros(
            pad_flat, dtype=input_ids.dtype, device=input_ids.device)
        pad_pos = torch.zeros(
            pad_flat, dtype=positions.dtype, device=positions.device)
        input_ids = torch.cat([input_ids, pad_ids], dim=0)
        positions = torch.cat([positions, pad_pos], dim=0)

        # Pad slot_mapping with -1 to skip KV writes for padded queries
        slot_map = torch.cat(
            [context.slot_mapping,
             torch.full((pad_flat,), -1, dtype=context.slot_mapping.dtype, device=context.slot_mapping.device)]
        )

        # Pad block_tables/context_lens by repeating the last real row
        bt = context.block_tables
        cl = context.context_lens
        pad_bt = bt[orig_B - 1:orig_B].expand(pad_B, -1).contiguous()
        pad_cl = cl[orig_B - 1:orig_B].expand(pad_B).contiguous()
        bt = torch.cat([bt, pad_bt], dim=0)
        cl = torch.cat([cl, pad_cl], dim=0)

        # Set padded context for this replay
        set_context(is_prefill=False, slot_mapping=slot_map,
                    context_lens=cl, block_tables=bt)

        block_tables = bt
        context_lens = cl
        flat_batch_size = input_ids.size(0)  # == wrapper_bs * MQ_LEN
        B = wrapper_bs
    else:
        block_tables = context.block_tables
        context_lens = context.context_lens
        flat_batch_size = orig_flat
        B = orig_B

    if PROFILE:
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

    # in the case where we pad, we'll need cache_hits.shape[0] to match the padded batch size
    if cache_hits.shape[0] < B:
        cache_hits = torch.cat([cache_hits, torch.zeros(B - cache_hits.shape[0], device=cache_hits.device)])

    # PERFORMANCE: Step 0 -- precompute KV page metadata on CPU for all K steps.
    # CPU tensors let plan() skip its internal .to("cpu") GPU->CPU syncs.
    # For B<=8, CPU slicing also avoids GPU boolean indexing.
    if step == 0:
        cache["cu_seqlens_q_cpu"] = torch.arange(B + 1, dtype=torch.int32) * MQ_LEN
        context_lens_list = context_lens.tolist()
        cache["block_tables"] = block_tables
        block_size = model_runner.block_size
        cache["precomputed_kv"] = []
        cache["plan_cpu_args"] = []

        if B <= 8:
            # PERFORMANCE: CPU-only kv_indices via slicing (no GPU boolean indexing)
            for s in range(K):
                step_cls = [int(cl) + s * MQ_LEN for cl in context_lens_list]
                step_counts = [(cl + block_size - 1) // block_size for cl in step_cls]
                if B == 1:
                    kv_indices_s = block_tables[0, :step_counts[0]]
                else:
                    kv_indices_s = torch.cat([block_tables[b, :step_counts[b]] for b in range(B)])
                cache["precomputed_kv"].append(kv_indices_s)
                kv_indptr_cpu = torch.zeros(B + 1, dtype=torch.int32)
                kv_indptr_cpu[1:] = torch.tensor(step_counts, dtype=torch.int32).cumsum(0)
                kv_lpl_cpu = torch.tensor(
                    [cl % block_size if cl % block_size != 0 else block_size for cl in step_cls],
                    dtype=torch.int32)
                cache["plan_cpu_args"].append((kv_indptr_cpu, kv_lpl_cpu))
        else:
            # Large batch: GPU boolean indexing for kv_indices, CPU tensors for plan args
            bt_upcast = torch.arange(block_tables.size(1), device=block_tables.device)[None, :]
            step_offsets = torch.arange(K + 2, device=context_lens.device) * MQ_LEN
            all_step_cls = context_lens.unsqueeze(1) + step_offsets.unsqueeze(0)
            all_counts = (all_step_cls + block_size - 1) // block_size
            all_masks = bt_upcast.unsqueeze(1) < all_counts.unsqueeze(2)
            for s in range(K):
                cache["precomputed_kv"].append(block_tables[all_masks[:, s, :]])
                step_cls = [int(cl) + s * MQ_LEN for cl in context_lens_list]
                step_counts = [(cl + block_size - 1) // block_size for cl in step_cls]
                kv_indptr_cpu = torch.zeros(B + 1, dtype=torch.int32)
                kv_indptr_cpu[1:] = torch.tensor(step_counts, dtype=torch.int32).cumsum(0)
                kv_lpl_cpu = torch.tensor(
                    [cl % block_size if cl % block_size != 0 else block_size for cl in step_cls],
                    dtype=torch.int32)
                cache["plan_cpu_args"].append((kv_indptr_cpu, kv_lpl_cpu))

        # CPU mask precompute: build all K packed masks using numpy at step 0.
        # Eliminates per-step get_custom_mask (GPU) + segment_packbits + GPU->CPU syncs.
        cache_hits_list = cache_hits[:B].tolist()

        if "glue_hit_np" not in cache:
            _fol = model_runner.config.fan_out_list
            _fol_miss = model_runner.config.fan_out_list_miss
            _tril = np.tril(np.ones((K + 1, K + 1), dtype=np.uint8))
            cache["glue_hit_np"] = np.repeat(_tril, _fol, axis=0)
            cache["glue_miss_np"] = np.repeat(_tril, _fol_miss, axis=0)

        _glue_hit = cache["glue_hit_np"]
        _glue_miss = cache["glue_miss_np"]
        _rows_np = np.arange(MQ_LEN)

        cache["cpu_packed_masks"] = []
        cache["cpu_packed_indptrs"] = []

        for s in range(K):
            ttl_added_s = (s + 1) * MQ_LEN + (K + 1)
            packed_segs = []
            seg_packed_sizes = []

            for b in range(B):
                cols_b = int(context_lens_list[b]) + s * MQ_LEN
                prefix_len_b = cols_b - ttl_added_s

                mask_b = np.zeros((MQ_LEN, cols_b), dtype=np.uint8)
                mask_b[:, :prefix_len_b] = 1
                glue = _glue_hit if int(cache_hits_list[b]) == 1 else _glue_miss
                mask_b[:, prefix_len_b:prefix_len_b + K + 1] = glue
                diag_start = prefix_len_b + K + 1
                for blk in range(s + 1):
                    mask_b[_rows_np, diag_start + blk * MQ_LEN + _rows_np] = 1

                packed = np.packbits(mask_b.ravel(), bitorder='little')
                packed_segs.append(packed)
                seg_packed_sizes.append(len(packed))

            full_packed = np.concatenate(packed_segs) if B > 1 else packed_segs[0]
            indptr = np.zeros(B + 1, dtype=np.int32)
            indptr[1:] = np.cumsum(seg_packed_sizes)

            cache["cpu_packed_masks"].append(
                torch.from_numpy(full_packed.copy()).to(model_runner.device, non_blocking=True))
            cache["cpu_packed_indptrs"].append(
                torch.from_numpy(indptr.copy()).to(model_runner.device, non_blocking=True))

        # Pre-transfer KV metadata to GPU (eliminates per-step pageable H2D transfers)
        cache["qo_indptr_gpu"] = cache["cu_seqlens_q_cpu"].to(model_runner.device, non_blocking=True)
        cache["kv_indptr_gpu"] = []
        cache["kv_lpl_gpu"] = []
        cache["kv_lens_gpu"] = []
        for s in range(K):
            ki, kl = cache["plan_cpu_args"][s]
            cache["kv_indptr_gpu"].append(ki.to(model_runner.device, non_blocking=True))
            cache["kv_lpl_gpu"].append(kl.to(model_runner.device, non_blocking=True))
            kv_lens = ((ki[1:] - ki[:-1] - 1) * model_runner.block_size + kl).to(torch.int32)
            cache["kv_lens_gpu"].append(kv_lens.to(model_runner.device, non_blocking=True))

    if PROFILE:
        end_time.record()
        torch.cuda.synchronize()
        precompute_time = start_time.elapsed_time(end_time)
        start_time.record()

    # Use precomputed CPU-packed masks (built at step 0)
    if PROFILE_DRAFT:
        _ev_mask0 = torch.cuda.Event(enable_timing=True); _ev_mask0.record()

    kv_indices = cache["precomputed_kv"][step]
    kv_indptr_cpu, kv_lpl_cpu = cache["plan_cpu_args"][step]
    qo_indptr_cpu = cache["cu_seqlens_q_cpu"]

    packed_mask = cache["cpu_packed_masks"][step]
    packed_indptr = cache["cpu_packed_indptrs"][step]
    wrapper._custom_mask_buf[:len(packed_mask)].copy_(packed_mask, non_blocking=True)
    wrapper._mask_indptr_buf.copy_(packed_indptr, non_blocking=True)

    # GPU-to-GPU copies from pre-transferred tensors (no pageable H2D)
    wrapper._qo_indptr_buf.copy_(cache["qo_indptr_gpu"], non_blocking=True)
    wrapper._paged_kv_indptr_buf.copy_(cache["kv_indptr_gpu"][step], non_blocking=True)
    wrapper._paged_kv_last_page_len_buf.copy_(cache["kv_lpl_gpu"][step], non_blocking=True)
    wrapper._paged_kv_indices_buf[:len(kv_indices)].copy_(kv_indices, non_blocking=True)

    total_num_rows = int(qo_indptr_cpu[-1].item())
    wrapper._kv_lens_buffer[:len(kv_indptr_cpu) - 1].copy_(cache["kv_lens_gpu"][step], non_blocking=True)

    # Event-based sync: only wait for this stream's copies, not all CUDA streams.
    global _plan_event
    if _plan_event is None:
        _plan_event = torch.cuda.Event()
    _plan_event.record()
    _plan_event.synchronize()

    if PROFILE_DRAFT:
        _ev_plan0 = torch.cuda.Event(enable_timing=True); _ev_plan0.record()

    plan_args = [
        wrapper._float_workspace_buffer, wrapper._int_workspace_buffer,
        wrapper._pin_memory_int_workspace_buffer,
        qo_indptr_cpu, kv_indptr_cpu, cache["kv_lens_gpu"][step],
        wrapper._max_total_num_rows or total_num_rows,
        B, model_runner.hf_config.num_attention_heads,
        model_runner.hf_config.num_key_value_heads,
        model_runner.block_size, wrapper.is_cuda_graph_enabled,
        model_runner.hf_config.head_dim, model_runner.hf_config.head_dim,
        False, -1,
    ]
    if wrapper._backend == "fa2":
        plan_args.extend([-1, False, 0])  # fixed_split_size, disable_split_kv, num_colocated_ctas
    wrapper._plan_info = wrapper._cached_module.plan(*plan_args)

    if PROFILE_DRAFT:
        _ev_plan1 = torch.cuda.Event(enable_timing=True); _ev_plan1.record()

    if PROFILE:
        end_time.record()
        torch.cuda.synchronize()
        plan_time = start_time.elapsed_time(end_time)
        start_time.record()

    # Copy inputs/context into graph buffers for padded size
    graph_vars["input_ids"][:flat_batch_size] = input_ids
    graph_vars["positions"][:flat_batch_size] = positions
    graph_vars["slot_mapping"][:flat_batch_size] = get_context().slot_mapping
    graph_vars["context_lens"][:B] = context_lens
    if hidden_states is not None and "hidden_states" in graph_vars:
        if hidden_states.shape[0] < flat_batch_size:
            # Pad hidden_states to match padded batch
            pad_n = flat_batch_size - hidden_states.shape[0]
            hidden_states = torch.cat([hidden_states, torch.zeros(pad_n, hidden_states.shape[1], dtype=hidden_states.dtype, device=hidden_states.device)])
        graph_vars["hidden_states"][:flat_batch_size] = hidden_states
    if step == 0:
        graph_vars["block_tables"][:B, :block_tables.size(1)] = block_tables

    if PROFILE:
        end_time.record()
        torch.cuda.synchronize()
        buffer_prep_time = start_time.elapsed_time(end_time)
        start_time.record()

    if PROFILE_DRAFT:
        _ev_replay0 = torch.cuda.Event(enable_timing=True); _ev_replay0.record()

    graph.replay()

    if PROFILE_DRAFT:
        _ev_replay1 = torch.cuda.Event(enable_timing=True); _ev_replay1.record()
        _draft_events.append((step, "mask+buf", _ev_mask0, _ev_plan0))
        _draft_events.append((step, "plan", _ev_plan0, _ev_plan1))
        _draft_events.append((step, "replay", _ev_replay0, _ev_replay1))

    if PROFILE:
        end_time.record()
        torch.cuda.synchronize()
        replay_time = start_time.elapsed_time(end_time)

    # Extract logits from graph_vars instead of computing them separately
    logits_all = graph_vars["logits"][:flat_batch_size]

    if PROFILE:
        print(f"[cuda_graph_helpers.run_fi_tree_decode_cudagraph] step {step}: precompute={precompute_time:.3f}ms, plan={plan_time:.3f}ms, buffer={buffer_prep_time:.3f}ms, replay={replay_time:.3f}ms", flush=True)

    logits_out = logits_all[:orig_flat]
    # EAGLE draft: also return prenorm (outputs) for self-conditioning
    if "hidden_states" in graph_vars:
        prenorm = graph_vars["outputs"][:orig_flat]
        return logits_out, prenorm
    return logits_out


## CAPTURE CUDAGRAPHS
@torch.inference_mode()
def capture_cudagraph(model_runner):
    config = model_runner.config
    hf_config = config.hf_config
    max_seqs = min(model_runner.config.max_num_seqs, 512)
    if model_runner.config.speculate and model_runner.config.draft_async and model_runner.is_draft:
        N = max_seqs * (model_runner.config.speculate_k + 1) * \
            model_runner.config.async_fan_out
        max_bs = N * (model_runner.config.speculate_k + 1)
    else:
        max_bs = max_seqs + 1
    max_num_blocks = (config.max_model_len +
                      model_runner.block_size - 1) // model_runner.block_size
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)

    graph_bs_list = [1, 2, 4, 8] + \
        list(range(16, max_bs + 1, 16))
    if max_bs % 16 != 0:
        graph_bs_list.append(max_bs)

    # make sure N is in graph_bs
    if model_runner.config.speculate and model_runner.config.draft_async and model_runner.is_draft:
        N = max_seqs * (model_runner.config.speculate_k + 1) * \
            model_runner.config.async_fan_out
        tree_decode_bs = N
        if tree_decode_bs not in graph_bs_list:
            # Insert in the correct position to maintain sorted order
            insert_pos = 0
            for i, bs in enumerate(graph_bs_list):
                if bs > tree_decode_bs:
                    insert_pos = i
                    break
                insert_pos = i + 1
            graph_bs_list.insert(insert_pos, tree_decode_bs)

    graphs = {}
    graph_pool = None

    is_jit = (model_runner.config.speculate and model_runner.config.draft_async and model_runner.is_draft)

    # Eagle models need special handling during CUDA graph capture
    is_eagle_draft = config.use_eagle and model_runner.is_draft
    is_eagle_target = config.use_eagle and not model_runner.is_draft
    hidden_states = None
    if is_eagle_draft:
        # Use hidden_size (d_model_draft) so CG captures the pass-through branch in Eagle3DraftForCausalLM.forward()
        # All callers project target acts via fc() BEFORE passing to CG
        hidden_states = torch.zeros(max_bs, hf_config.hidden_size,
                                    dtype=hf_config.torch_dtype, device=input_ids.device)

    for bs in reversed(graph_bs_list):
        graph = torch.cuda.CUDAGraph()
        set_context(
            False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], is_jit=is_jit)
        if is_eagle_draft:
            outputs[:bs] = model_runner.model(
                input_ids[:bs], positions[:bs], hidden_states[:bs])    # warmup
        elif is_eagle_target:
            out, _ = model_runner.model(
                input_ids[:bs], positions[:bs])    # warmup
            outputs[:bs] = out
        else:
            outputs[:bs] = model_runner.model(
                input_ids[:bs], positions[:bs])    # warmup
        with torch.cuda.graph(graph, graph_pool):
            if is_eagle_draft:
                outputs[:bs] = model_runner.model(
                    input_ids[:bs], positions[:bs], hidden_states[:bs])    # capture
            elif is_eagle_target:
                out, _ = model_runner.model(
                    input_ids[:bs], positions[:bs])    # capture
                outputs[:bs] = out
            else:
                outputs[:bs] = model_runner.model(
                    input_ids[:bs], positions[:bs])    # capture
        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
    )
    if hidden_states is not None:
        graph_vars["hidden_states"] = hidden_states

    return graph_vars, graph_pool, graphs, graph_bs_list


@torch.inference_mode()
def capture_verify_cudagraph(model_runner):
    config = model_runner.config
    # assert not model_runner.is_draft, "ERROR in capture_verify_cudagraph: verify path only supported for target model"
    hf_config = config.hf_config
    max_bs = min(model_runner.config.max_num_seqs, 512)
    k_plus_1 = model_runner.config.speculate_k + 1

    is_eagle_target = config.use_eagle and not model_runner.is_draft

    # For verify, we need to handle k+1 tokens per sequence, and use cu_seqlens_q and max_seqlen_q
    input_ids = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    positions = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs * k_plus_1, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(
        max_bs, model_runner.max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs * k_plus_1, hf_config.hidden_size)
    cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)

    # Eagle target: also capture eagle_acts from model forward
    eagle_acts = None
    if is_eagle_target:
        # eagle_acts has shape [num_tokens, 3 * hidden_size] for 3 layers
        eagle_acts = torch.zeros(max_bs * k_plus_1, 3 * hf_config.hidden_size,
                                  dtype=hf_config.torch_dtype)

    base = [1, 2, 4, 8]
    dynamic = list(range(16, max_bs+1, 16))
    all_b = base + dynamic
    if max_bs not in all_b:
        all_b.append(max_bs)
    all_b.sort()
    all_N = [b for b in all_b if b <= max_bs]

    graphs = {}
    graph_pool = None

    for bs in reversed(all_N):
        graph = torch.cuda.CUDAGraph()
        # For verify, each sequence is length K+1, so seqlen_q is [K+1]*bs
        seqlen_q = torch.full((bs,), k_plus_1, dtype=torch.int32)
        cu = cu_seqlens_q[:bs + 1]
        cu.zero_()
        cu[1:].copy_(torch.cumsum(seqlen_q, 0))
        context_lens[:bs] = seqlen_q

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping[:bs * k_plus_1],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
            cu_seqlens_q=cu,
            max_seqlen_q=k_plus_1,
        )

        # warmup
        model_out = model_runner.model(
            input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])
        if isinstance(model_out, tuple):
            outputs[:bs * k_plus_1] = model_out[0]
            if eagle_acts is not None:
                eagle_acts[:bs * k_plus_1] = model_out[1]
        else:
            outputs[:bs * k_plus_1] = model_out
        with torch.cuda.graph(graph, graph_pool):
            # capture
            model_out = model_runner.model(
                input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])
            if isinstance(model_out, tuple):
                outputs[:bs * k_plus_1] = model_out[0]
                if eagle_acts is not None:
                    eagle_acts[:bs * k_plus_1] = model_out[1]
            else:
                outputs[:bs * k_plus_1] = model_out

        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu_seqlens_q,
        outputs=outputs,
    )
    if eagle_acts is not None:
        graph_vars["eagle_acts"] = eagle_acts

    return graph_vars, graph_pool, graphs, all_N


@torch.inference_mode()
def run_glue_decode_cudagraph(model_runner, input_ids, positions, last_only, graph_vars, hidden_states=None):
    """Run EAGLE glue decode with FA causal + varlen cu_seqlens_q. No padding within sequences."""
    context = get_context()
    K = model_runner.config.speculate_k
    two_kp1 = 2 * K + 1
    orig_flat = input_ids.size(0)
    orig_B = context.context_lens.size(0)
    dev = input_ids.device

    wrapper_bs = next(
        x for x in model_runner.graph_bs_list["glue_decode"] if x >= orig_B)
    graph = model_runner.graphs["glue_decode"][wrapper_bs]
    max_flat = wrapper_bs * two_kp1

    # Zero all non-output graph vars
    for k, v in graph_vars.items():
        if k != "outputs":
            v.zero_()

    # Copy real data into graph buffers (orig_flat <= max_flat always)
    graph_vars["input_ids"][:orig_flat] = input_ids
    graph_vars["positions"][:orig_flat] = positions
    graph_vars["slot_mapping"][:orig_flat] = context.slot_mapping
    # Pad remaining flat slots with -1 slot_mapping (no KV write)
    if orig_flat < max_flat:
        graph_vars["slot_mapping"][orig_flat:max_flat] = -1

    graph_vars["context_lens"][:orig_B] = context.context_lens
    graph_vars["block_tables"][:orig_B, :context.block_tables.size(1)] = context.block_tables

    # cu_seqlens_q: real seqs, then ghost seqs (repeat last cumsum = 0-length queries)
    cu = context.cu_seqlens_q  # [orig_B + 1]
    graph_vars["cu_seqlens_q"][:orig_B + 1] = cu
    if wrapper_bs > orig_B:
        # Ghost seqs get 0-length queries
        graph_vars["cu_seqlens_q"][orig_B + 1:wrapper_bs + 1] = cu[-1]
        # Ghost seqs need valid block_tables/context_lens (copy last real seq)
        pad_B = wrapper_bs - orig_B
        graph_vars["context_lens"][orig_B:wrapper_bs] = context.context_lens[orig_B - 1]
        graph_vars["block_tables"][orig_B:wrapper_bs] = context.block_tables[orig_B - 1]

    if hidden_states is not None and "eagle_hidden_states" in graph_vars:
        graph_vars["eagle_hidden_states"][:orig_flat] = hidden_states

    graph.replay()

    outputs = graph_vars["outputs"][:orig_flat]
    logits = model_runner.model.compute_logits(outputs, last_only)
    if "eagle_hidden_states" in graph_vars:
        return logits, outputs
    return logits


@torch.inference_mode()
def capture_glue_decode_cudagraph(model_runner):
    """Capture CG for EAGLE glue decode: FA causal + varlen cu_seqlens_q, max flat = B*(2K+1)."""
    config = model_runner.config
    hf_config = config.hf_config
    max_bs = min(config.max_num_seqs, 512)
    K = config.speculate_k
    two_kp1 = 2 * K + 1
    max_flat = max_bs * two_kp1
    max_num_blocks = (config.max_model_len + model_runner.block_size - 1) // model_runner.block_size

    input_ids = torch.zeros(max_flat, dtype=torch.int64, device=model_runner.device)
    positions = torch.zeros(max_flat, dtype=torch.int64, device=model_runner.device)
    slot_mapping = torch.zeros(max_flat, dtype=torch.int32, device=model_runner.device)
    context_lens = torch.full((max_bs,), config.max_model_len, dtype=torch.int32, device=model_runner.device)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=model_runner.device)
    outputs = torch.empty(max_flat, hf_config.hidden_size, device=model_runner.device)
    cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32, device=model_runner.device)

    eagle_hs = None
    if config.use_eagle and model_runner.is_draft:
        eagle_hs = torch.zeros(max_flat, hf_config.hidden_size, dtype=hf_config.torch_dtype, device=model_runner.device)

    graph_bs_list = [1]
    for bs in [2, 4, 8] + list(range(16, max_bs + 1, 16)):
        if bs <= max_bs:
            graph_bs_list.append(bs)
    if max_bs not in graph_bs_list:
        graph_bs_list.append(max_bs)
    graph_bs_list.sort()

    graphs = {}
    graph_pool = None

    print(f'[cuda_graph_helpers.capture_glue_decode_cudagraph] Capturing for bs={graph_bs_list}', flush=True)

    for bs in reversed(graph_bs_list):
        graph = torch.cuda.CUDAGraph()
        flat = bs * two_kp1

        # Uniform cu_seqlens_q for capture (each seq gets 2K+1 queries)
        seqlen_q = torch.full((bs,), two_kp1, dtype=torch.int32, device=model_runner.device)
        cu = cu_seqlens_q[:bs + 1]
        cu.zero_()
        cu[1:].copy_(torch.cumsum(seqlen_q, 0))

        set_context(
            is_prefill=False,
            cu_seqlens_q=cu,
            max_seqlen_q=two_kp1,
            slot_mapping=slot_mapping[:flat],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
        )

        if eagle_hs is not None:
            outputs[:flat] = model_runner.model(input_ids[:flat], positions[:flat], eagle_hs[:flat])
        else:
            outputs[:flat] = model_runner.model(input_ids[:flat], positions[:flat])

        with torch.cuda.graph(graph, graph_pool):
            if eagle_hs is not None:
                outputs[:flat] = model_runner.model(input_ids[:flat], positions[:flat], eagle_hs[:flat])
            else:
                outputs[:flat] = model_runner.model(input_ids[:flat], positions[:flat])

        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu_seqlens_q,
        outputs=outputs,
    )
    if eagle_hs is not None:
        graph_vars["eagle_hidden_states"] = eagle_hs

    return graph_vars, graph_pool, graphs, graph_bs_list


@torch.inference_mode()
def capture_fi_tree_decode_cudagraph(model_runner):
    config = model_runner.config
    hf_config = config.hf_config
    max_bs = min(model_runner.config.max_num_seqs, 512)
    K, F = model_runner.config.speculate_k, model_runner.config.async_fan_out
    # MQ_LEN = F * (K+1)
    MQ_LEN = sum(model_runner.config.fan_out_list)
    max_flat_batch_size = max_bs * MQ_LEN

    max_num_blocks = (config.max_model_len +
                      model_runner.block_size - 1) // model_runner.block_size
    input_ids = torch.zeros(max_flat_batch_size, dtype=torch.int64, device=model_runner.device)
    positions = torch.zeros(max_flat_batch_size, dtype=torch.int64, device=model_runner.device)
    slot_mapping = torch.zeros(max_flat_batch_size, dtype=torch.int32, device=model_runner.device)
    context_lens = torch.full((max_bs,), config.max_model_len, dtype=torch.int32, device=model_runner.device) # make sure these are consistent with our dummy example
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=model_runner.device)
    outputs = torch.empty(max_flat_batch_size, hf_config.hidden_size, device=model_runner.device)
    logits = torch.empty(max_flat_batch_size, hf_config.vocab_size, device=model_runner.device)

    # Create graph_bs_list to match what will be used in cudagraph_helpers.py
    graph_bs_list = [1]
    for bs in [2, 4, 8] + list(range(16, max_bs + 1, 16)):
        if bs <= max_bs:
            graph_bs_list.append(bs)
    if max_bs not in graph_bs_list:
        graph_bs_list.append(max_bs)
    graph_bs_list.sort()

    graphs = {}
    graph_pool = None

    # Eagle draft needs hidden_states for forward (d_model_draft, NOT 3*d_model_target)
    # All callers project target acts via fc() BEFORE passing to CG
    # MUST be outside the for-loop so all graphs share the same tensor
    fi_hidden_states = None
    if config.use_eagle and model_runner.is_draft:
        fi_hidden_states = torch.zeros(max_flat_batch_size, hf_config.hidden_size,
                                       dtype=hf_config.torch_dtype, device=model_runner.device)

    print(f'[cuda_graph_helpers.capture_fi_tree_decode_cudagraph] About to capture FI cudagraphs for bs={graph_bs_list}', flush=True)

    for bs in reversed(graph_bs_list):
        graph = torch.cuda.CUDAGraph()

        # Build a self-consistent fake plan for capture:
        # - q_len = MQ_LEN for each request
        # - k_len = max_model_len for each request (use maximum context length)

        cu_seqlens_q = torch.arange(
            bs + 1, dtype=torch.int32, device=model_runner.device) * MQ_LEN
        # Use max_num_blocks pages per request for maximum context length
        kv_indptr = torch.arange(
            bs + 1, dtype=torch.int32, device=model_runner.device) * max_num_blocks
        kv_indices = torch.zeros(int(
            kv_indptr[-1].item()), dtype=torch.int32, device=model_runner.device)  # page ids (dummy)
        # Last page length for max model len context
        last_page_len = config.max_model_len % model_runner.block_size
        if last_page_len == 0:
            last_page_len = model_runner.block_size
        kv_last_page_len = torch.full(
            (bs,), last_page_len, dtype=torch.int32, device=model_runner.device)
        custom_mask = torch.ones(bs * MQ_LEN * config.max_model_len,
                                 dtype=torch.bool, device=model_runner.device)

        # Set the fi_tensors buffers with our fake data
        model_runner.prefill_wrappers[bs].plan(
            cu_seqlens_q,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            hf_config.num_attention_heads,
            hf_config.num_key_value_heads,
            hf_config.head_dim,
            model_runner.block_size,
            custom_mask=custom_mask,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

        # Set minimal context needed for run
        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping[:bs * MQ_LEN],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs]
        )

        # Warmup run
        if fi_hidden_states is not None:
            outputs[:bs * MQ_LEN] = model_runner.model(
                input_ids[:bs * MQ_LEN], positions[:bs * MQ_LEN], fi_hidden_states[:bs * MQ_LEN])
        else:
            outputs[:bs * MQ_LEN] = model_runner.model(
                input_ids[:bs * MQ_LEN], positions[:bs * MQ_LEN])
        logits[:bs * MQ_LEN] = model_runner.model.compute_logits(outputs[:bs * MQ_LEN], False)

        # Capture both model run and logits computation
        with torch.cuda.graph(graph, graph_pool):
            if fi_hidden_states is not None:
                outputs[:bs * MQ_LEN] = model_runner.model(
                    input_ids[:bs * MQ_LEN], positions[:bs * MQ_LEN], fi_hidden_states[:bs * MQ_LEN])
            else:
                outputs[:bs * MQ_LEN] = model_runner.model(input_ids[:bs * MQ_LEN], positions[:bs * MQ_LEN])
            logits[:bs * MQ_LEN] = model_runner.model.compute_logits(outputs[:bs * MQ_LEN], False)

        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph

        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        context_lens=context_lens,
        outputs=outputs,
        logits=logits,
    )
    if fi_hidden_states is not None:
        graph_vars["hidden_states"] = fi_hidden_states

    return graph_vars, graph_pool, graphs, graph_bs_list
