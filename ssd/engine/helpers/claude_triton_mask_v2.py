"""
Optimized Triton kernel for tree decode mask materialization - Version 2.

Optimizations: fused computation, better memory coalescing,
specialized kernels for common cases, no Python-side loops.
"""

import torch
import triton
import triton.language as tl
from typing import List, Optional, Tuple
import math


# =============================================================================
# Optimized Triton Kernels
# =============================================================================

@triton.jit
def _mask_kernel_fused_v2(
    output_ptr,
    context_lens_ptr,
    batch_offsets_ptr,
    B,
    K: tl.constexpr,
    F: tl.constexpr,
    step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Highly optimized fused kernel for uniform fan-out mask generation.

    Memory layout: output is [sum_b(MQ_LEN * ctx_lens[b])] flattened.
    Each block processes BLOCK_SIZE consecutive elements.
    """
    MQ_LEN: tl.constexpr = F * (K + 1)
    glue_cols: tl.constexpr = K + 1
    diag_cols: tl.constexpr = (step + 1) * MQ_LEN
    fixed_cols: tl.constexpr = glue_cols + diag_cols

    # 2D grid: (batch_idx, block_within_batch)
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    # Early exit for invalid batches
    if batch_idx >= B:
        return

    # Load batch-specific data
    ctx_len = tl.load(context_lens_ptr + batch_idx)
    batch_offset = tl.load(batch_offsets_ptr + batch_idx)

    total_elements = MQ_LEN * ctx_len
    block_start = block_idx * BLOCK_SIZE

    # Early exit if this block is beyond the batch
    if block_start >= total_elements:
        return

    prefix_len = ctx_len - fixed_cols

    # Generate element indices for this block
    elem_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    valid_mask = elem_offsets < total_elements

    # Compute 2D coordinates efficiently
    # For row-major layout: row = idx // cols, col = idx % cols
    row = elem_offsets // ctx_len
    col = elem_offsets % ctx_len

    # =================================================================
    # Compute mask values using vectorized conditionals
    # =================================================================

    # Start with zeros
    values = tl.zeros([BLOCK_SIZE], dtype=tl.int8)

    # Region 1: Prefix (col < prefix_len) -> always 1
    is_prefix = col < prefix_len
    values = tl.where(is_prefix, tl.full([BLOCK_SIZE], 1, dtype=tl.int8), values)

    # Region 2: Glue (prefix_len <= col < prefix_len + glue_cols)
    # Lower triangular with row mapping: glue_row = row // F
    glue_col_idx = col - prefix_len
    is_in_glue = (col >= prefix_len) & (col < prefix_len + glue_cols)
    glue_row_idx = row // F  # Maps to 0..K
    is_glue_valid = glue_col_idx <= glue_row_idx
    values = tl.where(is_in_glue & is_glue_valid, tl.full([BLOCK_SIZE], 1, dtype=tl.int8), values)

    # Region 3: Diagonal (col >= prefix_len + glue_cols)
    # (step+1) identity matrices of size MQ_LEN x MQ_LEN
    diag_col_idx = col - prefix_len - glue_cols
    is_in_diag = col >= prefix_len + glue_cols
    # Position within MQ_LEN-sized blocks: diag_col_idx % MQ_LEN == row
    diag_pos = diag_col_idx % MQ_LEN
    is_diag_valid = diag_pos == row
    values = tl.where(is_in_diag & is_diag_valid, tl.full([BLOCK_SIZE], 1, dtype=tl.int8), values)

    # Store results with global offset
    global_indices = batch_offset + elem_offsets
    tl.store(output_ptr + global_indices, values, mask=valid_mask)


@triton.jit
def _mask_kernel_single_batch_v2(
    output_ptr,
    prefix_len,
    ctx_len,
    K: tl.constexpr,
    F: tl.constexpr,
    step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for single batch - avoids batch offset overhead.
    """
    MQ_LEN: tl.constexpr = F * (K + 1)
    glue_cols: tl.constexpr = K + 1

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    total_elements = MQ_LEN * ctx_len

    if block_start >= total_elements:
        return

    elem_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    valid_mask = elem_offsets < total_elements

    row = elem_offsets // ctx_len
    col = elem_offsets % ctx_len

    values = tl.zeros([BLOCK_SIZE], dtype=tl.int8)

    # Prefix
    is_prefix = col < prefix_len
    values = tl.where(is_prefix, tl.full([BLOCK_SIZE], 1, dtype=tl.int8), values)

    # Glue
    glue_col_idx = col - prefix_len
    is_in_glue = (col >= prefix_len) & (col < prefix_len + glue_cols)
    glue_row_idx = row // F
    is_glue_valid = glue_col_idx <= glue_row_idx
    values = tl.where(is_in_glue & is_glue_valid, tl.full([BLOCK_SIZE], 1, dtype=tl.int8), values)

    # Diagonal
    diag_col_idx = col - prefix_len - glue_cols
    is_in_diag = col >= prefix_len + glue_cols
    diag_pos = diag_col_idx % MQ_LEN
    is_diag_valid = diag_pos == row
    values = tl.where(is_in_diag & is_diag_valid, tl.full([BLOCK_SIZE], 1, dtype=tl.int8), values)

    tl.store(output_ptr + elem_offsets, values, mask=valid_mask)


@triton.jit
def _mask_kernel_nonuniform_fanout(
    output_ptr,
    context_lens_ptr,
    batch_offsets_ptr,
    cache_hits_ptr,
    fan_out_cumsum_ptr,      # [2, K+2] - row 0 for hit, row 1 for miss
    B,
    K: tl.constexpr,
    MQ_LEN: tl.constexpr,
    step: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel supporting non-uniform fan-out with cache hit awareness.
    Uses precomputed cumulative fan-outs for efficient row-to-group mapping.
    """
    glue_cols: tl.constexpr = K + 1
    diag_cols = (step + 1) * MQ_LEN
    fixed_cols = glue_cols + diag_cols

    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    if batch_idx >= B:
        return

    ctx_len = tl.load(context_lens_ptr + batch_idx)
    batch_offset = tl.load(batch_offsets_ptr + batch_idx)
    cache_hit = tl.load(cache_hits_ptr + batch_idx)

    total_elements = MQ_LEN * ctx_len
    block_start = block_idx * BLOCK_SIZE

    if block_start >= total_elements:
        return

    prefix_len = ctx_len - fixed_cols

    elem_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    valid_mask = elem_offsets < total_elements

    row = elem_offsets // ctx_len
    col = elem_offsets % ctx_len

    values = tl.zeros([BLOCK_SIZE], dtype=tl.int8)

    # Prefix
    is_prefix = col < prefix_len
    values = tl.where(is_prefix, tl.full([BLOCK_SIZE], 1, dtype=tl.int8), values)

    # Glue with non-uniform fan-out
    glue_col_idx = col - prefix_len
    is_in_glue = (col >= prefix_len) & (col < prefix_len + glue_cols)

    # Map row to glue_row using cumsum lookup
    # fan_out_cumsum[hit_idx, k] gives cumsum up to position k
    hit_idx = tl.where(cache_hit != 0, 0, 1)  # 0 for hit, 1 for miss

    # Find which group each row belongs to by checking cumsum boundaries
    glue_row_idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for k in range(K + 1):
        cumsum_k = tl.load(fan_out_cumsum_ptr + hit_idx * (K + 2) + k)
        cumsum_k1 = tl.load(fan_out_cumsum_ptr + hit_idx * (K + 2) + k + 1)
        in_group = (row >= cumsum_k) & (row < cumsum_k1)
        glue_row_idx = tl.where(in_group, k, glue_row_idx)

    is_glue_valid = glue_col_idx <= glue_row_idx
    values = tl.where(is_in_glue & is_glue_valid, tl.full([BLOCK_SIZE], 1, dtype=tl.int8), values)

    # Diagonal
    diag_col_idx = col - prefix_len - glue_cols
    is_in_diag = col >= prefix_len + glue_cols
    diag_pos = diag_col_idx % MQ_LEN
    is_diag_valid = diag_pos == row
    values = tl.where(is_in_diag & is_diag_valid, tl.full([BLOCK_SIZE], 1, dtype=tl.int8), values)

    global_indices = batch_offset + elem_offsets
    tl.store(output_ptr + global_indices, values, mask=valid_mask)


# =============================================================================
# Python Wrapper Functions - Optimized
# =============================================================================

class TritonMaskCacheV2:
    """Optimized cache for Triton mask computation."""

    def __init__(self):
        self.fan_out_cumsum: Optional[torch.Tensor] = None  # [2, K+2]
        self.cached_params: Optional[Tuple] = None
        self.MQ_LEN: int = 0

    def precompute(self, K: int, fan_out_list: List[int], fan_out_list_miss: List[int], device: torch.device):
        """Precompute cumulative fan-outs in a single tensor."""
        self.MQ_LEN = sum(fan_out_list)

        # Pack both hit and miss cumsums into [2, K+2] tensor
        self.fan_out_cumsum = torch.zeros((2, K + 2), dtype=torch.int32, device=device)
        self.fan_out_cumsum[0, 1:] = torch.tensor(fan_out_list, dtype=torch.int32, device=device).cumsum(0)
        self.fan_out_cumsum[1, 1:] = torch.tensor(fan_out_list_miss, dtype=torch.int32, device=device).cumsum(0)

        self.cached_params = (K, tuple(fan_out_list), tuple(fan_out_list_miss), device)


_cache_v2 = TritonMaskCacheV2()


def triton_get_mask_iter_i_v2(i: int, prefix_len: int, K: int, F: int, device: torch.device = None) -> torch.Tensor:
    """Optimized version of get_mask_iter_i."""
    if device is None:
        device = torch.device('cuda')

    MQ_LEN = F * (K + 1)
    ctx_len = prefix_len + (K + 1) + (i + 1) * MQ_LEN
    total_elements = MQ_LEN * ctx_len

    # Use int8 for memory efficiency, convert to bool at end
    output = torch.empty(total_elements, dtype=torch.int8, device=device)

    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _mask_kernel_single_batch_v2[grid](
        output,
        prefix_len,
        ctx_len,
        K, F, i,
        BLOCK_SIZE,
    )

    return output.view(MQ_LEN, ctx_len).to(torch.bool)


def triton_get_custom_mask_vectorized_v2(
    context_lens: torch.Tensor,
    step: int,
    K: int,
    F: int,
    B: int,
    device: torch.device,
) -> torch.Tensor:
    """Optimized vectorized mask generation."""
    MQ_LEN = F * (K + 1)
    glue_cols = K + 1
    diag_cols = (step + 1) * MQ_LEN
    fixed_cols = glue_cols + diag_cols

    # Compute batch offsets
    batch_sizes = MQ_LEN * context_lens
    batch_offsets = torch.zeros(B + 1, dtype=torch.int64, device=device)
    batch_offsets[1:] = batch_sizes.cumsum(0)
    total_elements = int(batch_offsets[-1].item())

    # Allocate output (int8 for efficiency)
    output = torch.empty(total_elements, dtype=torch.int8, device=device)

    # Determine optimal block size and grid
    BLOCK_SIZE = 1024
    max_batch_elements = int(batch_sizes.max().item())
    blocks_per_batch = (max_batch_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    grid = (B, blocks_per_batch)

    _mask_kernel_fused_v2[grid](
        output,
        context_lens,
        batch_offsets,
        B,
        K, F, step,
        BLOCK_SIZE,
    )

    return output.to(torch.bool)


def triton_get_custom_mask_cached_v2(
    context_lens: torch.Tensor,
    step: int,
    K: int,
    fan_out_list: List[int],
    fan_out_list_miss: List[int],
    B: int,
    device: torch.device,
    cache_hits: torch.Tensor,
) -> torch.Tensor:
    """Non-uniform fan-out mask with cache hit support."""
    global _cache_v2

    MQ_LEN = sum(fan_out_list)
    glue_cols = K + 1
    diag_cols = (step + 1) * MQ_LEN
    fixed_cols = glue_cols + diag_cols

    # Update cache if needed
    current_params = (K, tuple(fan_out_list), tuple(fan_out_list_miss), device)
    if _cache_v2.cached_params != current_params:
        _cache_v2.precompute(K, fan_out_list, fan_out_list_miss, device)

    # Compute batch offsets
    batch_sizes = MQ_LEN * context_lens
    batch_offsets = torch.zeros(B + 1, dtype=torch.int64, device=device)
    batch_offsets[1:] = batch_sizes.cumsum(0)
    total_elements = int(batch_offsets[-1].item())

    output = torch.empty(total_elements, dtype=torch.int8, device=device)

    BLOCK_SIZE = 1024
    max_batch_elements = int(batch_sizes.max().item())
    blocks_per_batch = (max_batch_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    grid = (B, blocks_per_batch)

    _mask_kernel_nonuniform_fanout[grid](
        output,
        context_lens,
        batch_offsets,
        cache_hits.to(torch.int32),
        _cache_v2.fan_out_cumsum,
        B,
        K, MQ_LEN, step,
        BLOCK_SIZE,
    )

    return output.to(torch.bool)


@torch.inference_mode()
def triton_get_custom_mask_v2(
    config,
    context_lens: torch.Tensor,
    step: int,
    K: int,
    F: int,
    B: int,
    device: torch.device,
    cache_hits: torch.Tensor,
) -> torch.Tensor:
    """
    Main entry point for optimized Triton mask computation.
    Drop-in replacement for get_custom_mask.
    """
    fan_out_list = config.fan_out_list
    fan_out_list_miss = config.fan_out_list_miss

    is_uniform = all(f == F for f in fan_out_list) and all(f == F for f in fan_out_list_miss)

    if is_uniform:
        return triton_get_custom_mask_vectorized_v2(context_lens, step, K, F, B, device)

    return triton_get_custom_mask_cached_v2(
        context_lens, step, K, fan_out_list, fan_out_list_miss, B, device, cache_hits
    )


# =============================================================================
# Testing Functions
# =============================================================================

def test_correctness_against_pytorch():
    """Comprehensive correctness tests."""
    import sys
    sys.path.insert(0, '/tmp/ssd-triton-mask/ssd_original')
    from ssd.engine.helpers.mask_helpers import get_mask_iter_i, get_custom_mask_vectorized

    device = torch.device('cuda')
    all_passed = True

    print("=" * 70)
    print("Testing V2 Triton Kernels Against PyTorch Reference")
    print("=" * 70)

    # Test 1: get_mask_iter_i
    print("\n[Test 1] get_mask_iter_i equivalence:")
    test_cases_iter = [
        (0, 100, 3, 4),
        (1, 50, 3, 4),
        (2, 200, 3, 4),
        (0, 100, 2, 3),
        (3, 150, 5, 2),
        (0, 10, 1, 2),
        (5, 500, 4, 3),
    ]

    for i, prefix_len, K, F in test_cases_iter:
        ref = get_mask_iter_i(i, prefix_len, K, F).to(device)
        triton_out = triton_get_mask_iter_i_v2(i, prefix_len, K, F, device)

        if torch.equal(ref, triton_out):
            print(f"  PASS: i={i}, prefix={prefix_len}, K={K}, F={F}")
        else:
            diff = (ref != triton_out).sum().item()
            print(f"  FAIL: i={i}, prefix={prefix_len}, K={K}, F={F} ({diff}/{ref.numel()} differ)")
            all_passed = False

    # Test 2: get_custom_mask_vectorized
    print("\n[Test 2] get_custom_mask_vectorized equivalence:")
    test_cases_vec = [
        (1, 3, 4, 0, 200),
        (1, 3, 4, 2, 500),
        (2, 3, 4, 1, 200),
        (4, 3, 4, 2, 300),
        (8, 3, 4, 3, 400),
        (16, 2, 3, 1, 250),
        (1, 5, 4, 4, 600),
        (32, 3, 4, 2, 500),
    ]

    for B, K, F, step, ctx_base in test_cases_vec:
        MQ_LEN = F * (K + 1)
        glue_cols = K + 1
        diag_cols = (step + 1) * MQ_LEN
        fixed_cols = glue_cols + diag_cols

        context_lens = torch.tensor(
            [ctx_base + i * 10 for i in range(B)],
            dtype=torch.int64,
            device=device
        )

        ref = get_custom_mask_vectorized(context_lens, step, K, F, B, device)
        triton_out = triton_get_custom_mask_vectorized_v2(context_lens, step, K, F, B, device)

        if torch.equal(ref, triton_out):
            print(f"  PASS: B={B}, K={K}, F={F}, step={step}, ctx~{ctx_base}")
        else:
            diff = (ref != triton_out).sum().item()
            print(f"  FAIL: B={B}, K={K}, F={F}, step={step} ({diff}/{ref.numel()} differ)")
            # Debug: find first mismatch
            mismatch_idx = torch.where(ref != triton_out)[0][0].item()
            print(f"        First mismatch at idx {mismatch_idx}: ref={ref[mismatch_idx]}, triton={triton_out[mismatch_idx]}")
            all_passed = False

    return all_passed


def benchmark_v2():
    """Comprehensive benchmarks comparing V2 Triton to PyTorch."""
    import sys
    sys.path.insert(0, '/tmp/ssd-triton-mask/ssd_original')
    from ssd.engine.helpers.mask_helpers import get_custom_mask_vectorized, _get_custom_mask_optimized, _precompute_mask_components
    import time

    device = torch.device('cuda')

    # Warmup
    print("\nWarming up GPU...")
    for _ in range(20):
        ctx = torch.tensor([500], dtype=torch.int64, device=device)
        _ = get_custom_mask_vectorized(ctx, 2, 3, 4, 1, device)
        _ = triton_get_custom_mask_vectorized_v2(ctx, 2, 3, 4, 1, device)
    torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print("Performance Benchmarks (100 iterations, times in milliseconds)")
    print("=" * 80)

    configs = [
        # (B, K, F, step, ctx_len) - various realistic scenarios
        (1, 3, 4, 2, 500),    # Single batch, medium context
        (1, 3, 4, 2, 1000),   # Single batch, longer context
        (1, 3, 4, 2, 2000),   # Single batch, long context
        (1, 3, 4, 2, 4096),   # Single batch, max context
        (4, 3, 4, 2, 500),    # Small batch
        (8, 3, 4, 2, 500),    # Medium batch
        (16, 3, 4, 2, 500),   # Larger batch
        (32, 3, 4, 2, 500),   # Large batch
        (1, 5, 3, 3, 500),    # Different K, F
        (4, 5, 3, 3, 500),
        (8, 2, 6, 2, 1000),   # High fan-out
    ]

    num_runs = 100

    print(f"\n{'Config':<35} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 80)

    for B, K, F, step, ctx_base in configs:
        context_lens = torch.tensor(
            [ctx_base + i * 10 for i in range(B)],
            dtype=torch.int64,
            device=device
        )

        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = get_custom_mask_vectorized(context_lens, step, K, F, B, device)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / num_runs * 1000

        # Benchmark Triton V2
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = triton_get_custom_mask_vectorized_v2(context_lens, step, K, F, B, device)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / num_runs * 1000

        speedup = pytorch_time / triton_time
        config_str = f"B={B:2d}, K={K}, F={F}, step={step}, ctx~{ctx_base}"
        print(f"{config_str:<35} {pytorch_time:<15.4f} {triton_time:<15.4f} {speedup:<10.2f}x")

    print("-" * 80)

    # Additional detailed timing analysis
    print("\n" + "=" * 80)
    print("CUDA Event Timing (microseconds, 1000 iterations)")
    print("=" * 80)

    # Use CUDA events for more precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    test_config = (1, 3, 4, 2, 500)  # Single batch typical case
    B, K, F, step, ctx_base = test_config
    context_lens = torch.tensor([ctx_base], dtype=torch.int64, device=device)

    num_iters = 1000

    # PyTorch timing
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iters):
        _ = get_custom_mask_vectorized(context_lens, step, K, F, B, device)
    end_event.record()
    torch.cuda.synchronize()
    pytorch_us = start_event.elapsed_time(end_event) * 1000 / num_iters

    # Triton timing
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iters):
        _ = triton_get_custom_mask_vectorized_v2(context_lens, step, K, F, B, device)
    end_event.record()
    torch.cuda.synchronize()
    triton_us = start_event.elapsed_time(end_event) * 1000 / num_iters

    print(f"\nSingle batch (B=1, K=3, F=4, step=2, ctx=500):")
    print(f"  PyTorch: {pytorch_us:.2f} μs")
    print(f"  Triton:  {triton_us:.2f} μs")
    print(f"  Speedup: {pytorch_us/triton_us:.2f}x")


if __name__ == "__main__":
    print("=" * 70)
    print("Triton Tree Decode Mask V2 - Tests and Benchmarks")
    print("=" * 70)

    if test_correctness_against_pytorch():
        print("\n" + "=" * 70)
        print("All correctness tests PASSED!")
        print("=" * 70)
        benchmark_v2()
    else:
        print("\n" + "=" * 70)
        print("Some tests FAILED - fix issues before benchmarking")
        print("=" * 70)
