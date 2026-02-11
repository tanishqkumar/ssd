"""
Correctness test: verify vectorized path (B>8) matches cached path (reference)
for both uniform and non-uniform fan_out_list.
Run on a GPU node: python test_unified_mask.py
"""
import sys
sys.path.insert(0, '/home/tkumar/ssd')

import torch
import time

from ssd.engine.helpers.mask_helpers import (
    get_custom_mask_cached,
    get_custom_mask_vectorized,
)

device = torch.device('cuda')

class MockConfig:
    def __init__(self, fan_out_list, fan_out_list_miss, max_model_len=8192):
        self.fan_out_list = fan_out_list
        self.fan_out_list_miss = fan_out_list_miss
        self.max_model_len = max_model_len


def test_correctness():
    """Compare vectorized vs cached (reference) for various configs at B>8."""
    test_cases = [
        # (name, fan_out_list_hit, fan_out_list_miss, K)
        ("uniform_f3_K6",   [3,3,3,3,3,3,3], [3,3,3,3,3,3,3], 6),
        ("nonunif_K6",      [3,3,3,3,3,3,3], [15,5,1,0,0,0,0], 6),
        ("uniform_f2_K4",   [2,2,2,2,2],     [2,2,2,2,2],     4),
        ("nonunif_K4",      [2,2,2,2,2],     [5,3,2,0,0],     4),
        ("uniform_f1_K6",   [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], 6),
        ("nonunif_f1_K6",   [1,1,1,1,1,1,1], [4,2,1,0,0,0,0], 6),
    ]

    B_values = [1, 2, 4, 8, 16, 32, 64]
    steps_to_test = lambda K: [0, K//2, K]

    total = 0
    passed = 0
    failed = 0

    for name, fol, folm, K in test_cases:
        F = fol[0]
        config = MockConfig(fol, folm)
        MQ_LEN = sum(fol)

        for B in B_values:
            for step in steps_to_test(K):
                for hit_pattern in ['all_hit', 'all_miss', 'mixed']:
                    total += 1
                    ttl_added = (K + 1) + (step + 1) * MQ_LEN
                    context_lens = torch.randint(
                        ttl_added + 50, ttl_added + 500, (B,), device=device)

                    if hit_pattern == 'all_hit':
                        cache_hits = torch.ones(B, device=device, dtype=torch.long)
                    elif hit_pattern == 'all_miss':
                        cache_hits = torch.zeros(B, device=device, dtype=torch.long)
                    else:
                        cache_hits = torch.randint(0, 2, (B,), device=device)

                    # Reference: cached path (loops over batch, always correct)
                    mask_ref = get_custom_mask_cached(
                        config, context_lens, step, K, F, B, device,
                        fan_out_list=fol, fan_out_list_miss=folm,
                        cache_hits=cache_hits)

                    # Vectorized path (being tested)
                    mask_vec = get_custom_mask_vectorized(
                        config, context_lens, step, K, B, device,
                        cache_hits=cache_hits)

                    if torch.equal(mask_ref, mask_vec):
                        passed += 1
                    else:
                        failed += 1
                        diff_count = (mask_ref != mask_vec).sum().item()
                        print(f"  FAIL {name} B={B} step={step} hits={hit_pattern} "
                              f"diffs={diff_count}/{mask_ref.numel()}")

    print(f"\nCorrectness: {passed}/{total} passed, {failed} failed")
    return failed == 0


def test_perf():
    """Benchmark vectorized vs cached at B>8."""
    K = 6
    configs = [
        ("uniform_f3",  [3,3,3,3,3,3,3], [3,3,3,3,3,3,3]),
        ("nonunif_f3",  [3,3,3,3,3,3,3], [15,5,1,0,0,0,0]),
        ("uniform_f1",  [1,1,1,1,1,1,1], [1,1,1,1,1,1,1]),
        ("nonunif_f1",  [1,1,1,1,1,1,1], [4,2,1,0,0,0,0]),
    ]
    B_values = [16, 32, 64]
    step = K
    warmup = 50
    iters = 200

    print(f"\n{'Config':<15} {'B':>4} {'Cached (us)':>12} {'Vectorized (us)':>15} {'Speedup':>8}")
    print("-" * 65)

    for cfg_name, fol, folm in configs:
        F = fol[0]
        config = MockConfig(fol, folm)
        MQ_LEN = sum(fol)

        for B in B_values:
            ttl_added = (K + 1) + (step + 1) * MQ_LEN
            context_lens = torch.randint(ttl_added + 50, ttl_added + 500, (B,), device=device)
            cache_hits = torch.randint(0, 2, (B,), device=device)

            for _ in range(warmup):
                get_custom_mask_cached(config, context_lens, step, K, F, B, device,
                                       fan_out_list=fol, fan_out_list_miss=folm,
                                       cache_hits=cache_hits)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                get_custom_mask_cached(config, context_lens, step, K, F, B, device,
                                       fan_out_list=fol, fan_out_list_miss=folm,
                                       cache_hits=cache_hits)
            torch.cuda.synchronize()
            cached_us = (time.perf_counter() - t0) / iters * 1e6

            for _ in range(warmup):
                get_custom_mask_vectorized(config, context_lens, step, K, B, device,
                                           cache_hits=cache_hits)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                get_custom_mask_vectorized(config, context_lens, step, K, B, device,
                                           cache_hits=cache_hits)
            torch.cuda.synchronize()
            vec_us = (time.perf_counter() - t0) / iters * 1e6

            speedup = cached_us / vec_us if vec_us > 0 else float('inf')
            print(f"{cfg_name:<15} {B:>4} {cached_us:>12.1f} {vec_us:>15.1f} {speedup:>7.2f}x")


if __name__ == '__main__':
    print("=" * 65)
    print("Correctness: vectorized (B>8) vs cached (reference)")
    print("=" * 65)
    ok = test_correctness()

    if ok:
        print("\nAll correctness tests passed!")
        test_perf()
    else:
        print("\nCorrectness failures — skipping perf test")
        sys.exit(1)
