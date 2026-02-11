# SSD Reference Test Outputs

Generated: 2026-02-10
Commit: `8f8d3bc` (sync-free + B>1 fixes + unified mask)
Cluster: research-secure (H100 80GB, 6 GPUs per node)

Prompt 1: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
Prompt 2: "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"

---

## 1. Llama 70B AR (4 GPUs, no speculation)

**Command:**
```
python -O bench/bench.py --llama --size 70 --gpus 4 --b 1 --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 54 tok/s |
| Total Throughput | 53.72 tok/s |

**Generations (temp=0):**
1. `**\n* [3]**Natalia sold 48 clips in April. She sold half as many in May, so she sold 48 + 2 = 24 clips in May. Altogether, she sold 48 + 24 = 72 clips.`
2. `(Round to the nearest cent.)\n## Step 1: Calculate the fraction of an hour that 50 minutes represents.\nSince there are 60 minutes in an hour, 50 min...`

---

## 2. Llama 70B Sync Spec K=6 f=3 (4 GPUs)

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 4 --spec --k 6 --f 3 --b 1 --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | (uses different output format) |

**Generations:** Same as AR (temp=0, deterministic).

---

## 3. Llama 70B Async Spec K=6 f=3 — Batch Size Sweep

### B=1 (ns=128)

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 128 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 270 tok/s |
| Avg draft step | 21.12 ms |

### B=1 (ns=8)

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 238 tok/s |
| Avg draft step | 23.30 ms |

### B=2

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 2 --jit --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 409 tok/s |
| Avg draft step | 24.88 ms |

### B=4

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 4 --jit --numseqs 16 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 681 tok/s |
| Avg draft step | 28.96 ms |

### B=8

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 8 --jit --numseqs 32 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 1166 tok/s |
| Avg draft step | 35.29 ms |

**Generations (all batch sizes, temp=0):**
1. `**\n* [3]**Natalia sold 48 clips in April. She sold half as many in May, so she sold 48 + 2 = 24 clips in May. Altogether, she sold 48 + 24 = 72 clips.`
2. `(Round to the nearest cent.)\n## Step 1: Calculate the fraction of an hour that 50 minutes represents.\nSince there are 60 minutes in an hour, 50 min...`

---

## 4. Llama 70B Async Spec — Temperature Sweep (B=1)

### temp=0

| Metric | Value |
|--------|-------|
| Decode Throughput | 238 tok/s |
| Avg draft step | 23.30 ms |

### temp=0.7

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 8 --output_len 512 --temp 0.7
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 229 tok/s |
| Avg draft step | 22.64 ms |

**Generations (temp=0.7, non-deterministic):**
1. `**\n## Step 1: Identify the number of clips sold in April\nNatalia sold 48 clips to her friends in April.\n\n## Step 2: Calculate the number of clips...`
2. `**\n\n## Step 1: Calculate the hourly wage in terms of minutes.\nSince there are 60 minutes in an hour, we need to calculate how much Wong earns per m...`

### temp=1.0

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 8 --output_len 512 --temp 1.0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 179 tok/s |
| Avg draft step | 23.11 ms |

---

## 5. Qwen 32B (4/5 GPUs)

### AR (4 GPUs)

**Command:**
```
python -O bench/bench.py --size 32 --gpus 4 --b 1 --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 89 tok/s |

### Sync Spec K=4 f=2 (4 GPUs)

**Command:**
```
python -O bench/bench.py --size 32 --draft 0.6 --gpus 4 --spec --k 4 --f 2 --b 1 --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 156 tok/s |

### Async Spec K=4 f=2 (5 GPUs)

**Command:**
```
python -O bench/bench.py --size 32 --draft 0.6 --gpus 5 --spec --async --k 4 --f 2 --b 1 --jit --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 195 tok/s |
| Avg draft step | 17.93 ms |

**Generations (Qwen, temp=0):**
1. `To find the total number of clips Natalia sold in April and May, we need to add the number of clips sold in April to the number sold in May.\n\n1. **...`
2. `To find out how much Weng earned, we need to calculate the amount she earns per minute and then multiply that by the number of minutes she worked.\n\...`

---

## 6. Llama 70B Ablations (B=1, K=6 f=3)

### NO-JIT (default cudagraphs)

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 237 tok/s |
| Avg draft step | 22.21 ms |

### EAGER (no CUDA graphs)

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --numseqs 8 --output_len 512 --temp 0 --eager
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 55 tok/s |
| Avg draft step | 91.19 ms |

### JIT

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 241 tok/s |
| Avg draft step | 22.89 ms |

### sampler_x=0.05, temp=1.3

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 8 --output_len 512 --temp 1.3 --x 0.05
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 128 tok/s |
| Avg draft step | 30.84 ms |

### sampler_x=0.25, temp=1.3

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 8 --output_len 512 --temp 1.3 --x 0.25
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 135 tok/s |
| Avg draft step | 30.59 ms |

### Custom fan_out_list: flh=[3,3,3,3,3,3,3] flm=[15,5,1,0,0,0,0]

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 8 --output_len 512 --temp 0 --flh 3 3 3 3 3 3 3 --flm 15 5 1 0 0 0 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 224 tok/s |
| Avg draft step | 24.73 ms |

**Generations (all ablations at temp=0):** Identical to Section 3 (same model, deterministic).

---

## 7. Unified Mask vs Legacy — A/B Comparison

Unified mask (`get_custom_mask_unified`) replaces the legacy two-path dispatch (cached for B<=8, vectorized for B>8).
Unit test: **324/324 passed** (all batch sizes, uniform + non-uniform fan-out, all hit/miss patterns).

### Uniform fan_out (flh=flm=[3,3,3,3,3,3,3])

| B | Legacy Decode (tok/s) | Unified Decode (tok/s) | Legacy Draft (ms) | Unified Draft (ms) | Node |
|---|----------------------|----------------------|-------------------|-------------------|------|
| 1 | 254 | 246 | 21.59 | 22.55 | RS-15 / RS-20 |
| 4 | 695 | 685 | 28.37 | 28.81 | RS-16 / RS-23 |
| 16 | (OOM) | 1900 | - | 43.84 | - / RS-32 |

### Non-uniform fan_out (flh=[3,3,3,3,3,3,3], flm=[15,5,1,0,0,0,0])

| B | Legacy Decode (tok/s) | Unified Decode (tok/s) | Legacy Draft (ms) | Unified Draft (ms) | Node |
|---|----------------------|----------------------|-------------------|-------------------|------|
| 1 | 227 | 193 | 24.37 | 28.81 | RS-15 / RS-20 |
| 4 | 630 | 592 | 30.84 | 32.86 | RS-16 / RS-23 |
| 16 | (assert crash) | 1642 | - | 50.72 | - / RS-32 |

**Key observations:**
- Uniform: Unified is ~3% slower at B=1 due to `flat_blocks_after_cat` overhead vs simple cached path
- Non-uniform: Unified is ~15% slower at B=1 due to glue fixup overhead
- B=16 non-uniform: **Only unified can run** (legacy asserts on uniform fan-out at B>8)
- Generations are identical across all paths

### Mask generation microbenchmark (isolated, not e2e)

| Config | B | Cached (us) | Unified (us) | Speedup |
|--------|---|------------|-------------|---------|
| uniform | 1 | 190 | 311 | 0.61x |
| uniform | 4 | 474 | 313 | 1.51x |
| uniform | 8 | 807 | 344 | 2.34x |
| uniform | 16 | 1553 | 402 | 3.86x |
| uniform | 32 | 2877 | 519 | 5.54x |
| uniform | 64 | 6446 | 1134 | 5.68x |
| nonuniform | 1 | 179 | 651 | 0.27x |
| nonuniform | 4 | 595 | 729 | 0.82x |
| nonuniform | 8 | 1052 | 669 | 1.57x |
| nonuniform | 16 | 1579 | 1032 | 1.53x |
| nonuniform | 32 | 2858 | 691 | 4.13x |
| nonuniform | 64 | 5509 | 926 | 5.95x |

---

## Summary Table

| Config | Decode tok/s | Notes |
|--------|-------------|-------|
| **Llama 70B AR** | 54 | 4 GPUs, baseline |
| **Llama 70B Sync Spec K=6 f=3** | ~172 | 4 GPUs |
| **Llama 70B Async K=6 f=3 B=1 ns=8** | 238 | 5 GPUs, JIT |
| **Llama 70B Async K=6 f=3 B=1 ns=128** | 270 | 5 GPUs, JIT |
| **Llama 70B Async K=6 f=3 B=2** | 409 | 5 GPUs, JIT |
| **Llama 70B Async K=6 f=3 B=4** | 681 | 5 GPUs, JIT |
| **Llama 70B Async K=6 f=3 B=8** | 1166 | 5 GPUs, JIT |
| **Llama 70B Async K=6 f=3 B=16** | 1900 | 5 GPUs, JIT (unified mask) |
| Llama 70B Async temp=0.7 | 229 | B=1, JIT |
| Llama 70B Async temp=1.0 | 179 | B=1, JIT |
| Llama 70B NO-JIT | 237 | B=1, ~same as JIT |
| Llama 70B EAGER | 55 | B=1, ~= AR speed |
| Llama 70B sampler_x=0.05 temp=1.3 | 128 | B=1, JIT |
| Llama 70B sampler_x=0.25 temp=1.3 | 135 | B=1, JIT |
| Llama 70B custom FOL | 224 | B=1, JIT, flm=[15,5,1,0,0,0,0] |
| **Qwen 32B AR** | 89 | 4 GPUs |
| **Qwen 32B Sync Spec K=4 f=2** | 156 | 4 GPUs |
| **Qwen 32B Async Spec K=4 f=2** | 195 | 5 GPUs, JIT |

---

*End of reference results.*
