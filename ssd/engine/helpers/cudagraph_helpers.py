import os
import torch
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

    graph.replay() # [1.35ms/1.85ms]

    # Extract outputs for the ORIGINAL batch size only
    outputs = graph_vars["outputs"][:orig_bs * k_plus_1]
    return model_runner.model.compute_logits(outputs, last_only)


@torch.inference_mode()
def run_decode_cudagraph(model_runner, input_ids, positions, last_only, graph_vars):
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

    if context.block_tables is not None:
        graph_vars["block_tables"][:flat_batch_size,
                                :context.block_tables.size(1)] = context.block_tables

    graph.replay() # 32b is 23ms, 0.6b is 1.6ms [qwen, tp=1]. 10ms on tp=4.


    outputs = graph_vars["outputs"][:flat_batch_size]
    if getattr(model_runner.config, 'verbose', False):
        print(f"[run_decode_cudagraph] outputs={tuple(outputs.shape)}", flush=True)

    return model_runner.model.compute_logits(outputs, last_only)


cache = {}

PROFILE = os.environ.get("SSD_PROFILE", "0") == "1"

@torch.inference_mode()
def run_fi_tree_decode_cudagraph(model_runner, input_ids, positions, last_only, graph_vars, step, cache_hits):
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

    if PROFILE:
        end_time.record()
        torch.cuda.synchronize()
        precompute_time = start_time.elapsed_time(end_time)
        start_time.record()

    # Sync-free plan: call C++ plan directly, skip Python plan()'s 3 GPU syncs
    kv_indices = cache["precomputed_kv"][step]
    kv_indptr_cpu, kv_lpl_cpu = cache["plan_cpu_args"][step]
    qo_indptr_cpu = cache["cu_seqlens_q_cpu"]

    custom_mask = get_custom_mask(
        model_runner.config, context_lens, step, K, F, B,
        device=model_runner.device, cache_hits=cache_hits)
    mask_indptr_cpu = _compute_page_mask_indptr_cpu(
        qo_indptr_cpu, kv_indptr_cpu, kv_lpl_cpu, model_runner.block_size)
    packed_mask, packed_indptr_gpu = _segment_packbits_no_sync(
        custom_mask.contiguous().view(-1), mask_indptr_cpu)

    wrapper._custom_mask_buf[:len(packed_mask)].copy_(packed_mask, non_blocking=True)
    wrapper._mask_indptr_buf.copy_(packed_indptr_gpu, non_blocking=True)
    wrapper._qo_indptr_buf.copy_(qo_indptr_cpu.to(model_runner.device, non_blocking=True), non_blocking=True)
    wrapper._paged_kv_indptr_buf.copy_(kv_indptr_cpu.to(model_runner.device, non_blocking=True), non_blocking=True)
    wrapper._paged_kv_last_page_len_buf.copy_(kv_lpl_cpu.to(model_runner.device, non_blocking=True), non_blocking=True)
    wrapper._paged_kv_indices_buf[:len(kv_indices)].copy_(kv_indices, non_blocking=True)

    kv_lens_cpu = ((kv_indptr_cpu[1:] - kv_indptr_cpu[:-1] - 1) * model_runner.block_size + kv_lpl_cpu).to(torch.int32)
    total_num_rows = int(qo_indptr_cpu[-1].item())
    wrapper._kv_lens_buffer[:len(kv_lens_cpu)].copy_(kv_lens_cpu, non_blocking=True)
    plan_args = [
        wrapper._float_workspace_buffer, wrapper._int_workspace_buffer,
        wrapper._pin_memory_int_workspace_buffer,
        qo_indptr_cpu, kv_indptr_cpu, kv_lens_cpu,
        wrapper._max_total_num_rows or total_num_rows,
        B, model_runner.hf_config.num_attention_heads,
        model_runner.hf_config.num_key_value_heads,
        model_runner.block_size, wrapper.is_cuda_graph_enabled,
        model_runner.hf_config.head_dim, model_runner.hf_config.head_dim,
        False, -1,
    ]
    if wrapper._backend == "fa2":
        plan_args.extend([-1, False])
    wrapper._plan_info = wrapper._cached_module.plan(*plan_args)

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
    if step == 0:
        graph_vars["block_tables"][:B, :block_tables.size(1)] = block_tables

    if PROFILE:
        end_time.record()
        torch.cuda.synchronize()
        buffer_prep_time = start_time.elapsed_time(end_time)
        start_time.record()

    graph.replay()

    if PROFILE:
        end_time.record()
        torch.cuda.synchronize()
        replay_time = start_time.elapsed_time(end_time)

    # Extract logits from graph_vars instead of computing them separately
    logits_all = graph_vars["logits"][:flat_batch_size]

    if PROFILE:
        print(f"[run_fi_tree_decode_cudagraph] step {step}: precompute={precompute_time:.3f}ms, plan={plan_time:.3f}ms, buffer={buffer_prep_time:.3f}ms, replay={replay_time:.3f}ms", flush=True)

    return logits_all[:orig_flat]


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
    for bs in reversed(graph_bs_list):
        graph = torch.cuda.CUDAGraph()
        set_context(
            False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], is_jit=is_jit)
        outputs[:bs] = model_runner.model(
            input_ids[:bs], positions[:bs])    # warmup
        with torch.cuda.graph(graph, graph_pool):
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

    return graph_vars, graph_pool, graphs, graph_bs_list


@torch.inference_mode()
def capture_verify_cudagraph(model_runner):
    config = model_runner.config
    # assert not model_runner.is_draft, "ERROR in capture_verify_cudagraph: verify path only supported for target model"
    hf_config = config.hf_config
    max_bs = min(model_runner.config.max_num_seqs, 512)
    k_plus_1 = model_runner.config.speculate_k + 1

    # For verify, we need to handle k+1 tokens per sequence, and use cu_seqlens_q and max_seqlen_q
    input_ids = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    positions = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs * k_plus_1, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(
        max_bs, model_runner.max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs * k_plus_1, hf_config.hidden_size)
    cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)

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
        outputs[:bs * k_plus_1] = model_runner.model(
            input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])
        with torch.cuda.graph(graph, graph_pool):
            # capture
            outputs[:bs * k_plus_1] = model_runner.model(
                input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])

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

    return graph_vars, graph_pool, graphs, all_N



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

    print(f'About to capture FI cudagraphs for bs={graph_bs_list}', flush=True)

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
        outputs[:bs * MQ_LEN] = model_runner.model(
            input_ids[:bs * MQ_LEN], positions[:bs * MQ_LEN])
        logits[:bs * MQ_LEN] = model_runner.model.compute_logits(outputs[:bs * MQ_LEN], False)

        # Capture both model run and logits computation
        with torch.cuda.graph(graph, graph_pool):
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

    return graph_vars, graph_pool, graphs, graph_bs_list
