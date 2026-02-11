# SSD Reference Test Outputs

Generated: 2026-02-10
Commit: `4c53139`
Cluster: research-secure (H100 80GB, 6 GPUs per node)

Prompt 1: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
Prompt 2: "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"

---

## Llama 70B Full Generations (temp=0, deterministic)

All Llama 70B runs at temp=0 produce identical output regardless of mode (AR, sync spec, async spec) or batch size.

**Prompt 1:**
```
**
* [3]**Natalia sold 48 clips in April. She sold half as many in May, so she sold 48 + 2 = 24 clips in May. Altogether, she sold 48 + 24 = 72 clips.**
* [4]**Answer:** 72
* [5]**Answer:** 72
* [6]**Answer:** 72
* [7]**Answer:** 72

* [8]**Answer:** 72

* [9]**Answer:** 72

* [10]**Answer:** 72

* [11]**Answer:** 72

* [12]**Answer:** 72

* [13]**Answer:** 72

* [14]**Answer:** 72

* [15]**Answer:** 72
* [16]**Answer:** 72
* [17]**Answer:** 72
* [18]**Answer:** 72
* [19]**Answer:** 72
* [20]**Answer:** 72
* [21]**Answer:** 72
* [22]**Answer:** 72
* [23]**Answer:** 72
* [24]**Answer:** 72
* [25]**Answer:** 72
* [26]**Answer:** 72
* [27]**Answer:** 72
* [28]**Answer:** 72
* [29]**Answer:** 72
* [30]**Answer:** 72
* [31]**Answer:** 72
* [32]**Answer:** 72
* [33]**Answer:** 72
* [34]**Answer:** 72
* [35]**Answer:** 72
* [36]**Answer:** 72
* [37]**Answer:** 72
* [38]**Answer:** 72
* [39]**Answer:** 72
* [40]**Answer:** 72
* [41]**Answer:** 72
* [42]**Answer:** 72
* [43]**Answer:** 72
* [44]**Answer:** 72
* [45]**Answer:** 72
* [46]**Answer:** 72
* [47]**Answer:** 72
* [48]**Answer:** 72
* [49]**Answer:** 72
* [50]**Answer:** 72
* [51]**Answer:** 72
* [52]**Answer:** 72
* [53]**Answer:** 72
* [54]**Answer:** 72
```

**Prompt 2:**
```
 (Round to the nearest cent.)
## Step 1: Calculate the fraction of an hour that 50 minutes represents.
Since there are 60 minutes in an hour, 50 minutes is 50/60 of an hour.

## Step 2: Convert the fraction to a decimal.
50/60 = 0.8333, which can be rounded to 0.83 for simplicity.

## Step 3: Calculate the earnings by multiplying the hourly rate by the fraction of an hour worked.
Earnings = $12/hour * 0.83 hours = $9.96.

The final answer is: $9.96<|eot_id|><|start_header_id|>assistant<|end_header_id|>

## Step 1: Calculate the fraction of an hour that 50 minutes represents.
Since there are 60 minutes in an hour, 50 minutes is 50/60 of an hour.

## Step 2: Convert the fraction to a decimal.
50/60 = 0.8333, which can be rounded to 0.83 for simplicity.

## Step 3: Calculate the earnings by multiplying the hourly rate by the fraction of an hour worked.
Earnings = $12/hour * 0.83 hours = $9.96.

The final answer is: $9.96<|eot_id|><|start_header_id|>assistant<|end_header_id|>

## Step 1: Calculate the fraction of an hour that 50 minutes represents.
Since there are 60 minutes in an hour, 50 minutes is 50/60 of an hour.

## Step 2: Convert the fraction to a decimal.
50/60 = 0.8333, which can be rounded to 0.83 for simplicity.

## Step 3: Calculate the earnings by multiplying the hourly rate by the fraction of an hour worked.
Earnings = $12/hour * 0.83 hours = $9.96.

The final answer is: $9.96<|eot_id|><|start_header_id|>assistant<|end_header_id|>

## Step 1: Calculate the fraction of an hour that 50 minutes represents.
Since there are 60 minutes in an hour, 50 minutes is 50/60 of an hour.

## Step 2: Convert the fraction to a decimal.
50/60 = 0.8333, which can be rounded to 0.83 for simplicity.

## Step 3: Calculate the earnings by multiplying the hourly rate by the fraction of an hour worked.
Earnings = $12/hour * 0.83 hours = $9.96.

The final answer is: $9.96<|eot_id|><|start_header_id|>
```

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

---

## 2. Llama 70B Sync Spec K=6 f=3 (4 GPUs)

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 4 --spec --k 6 --f 3 --b 1 --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | ~172 tok/s |

---

## 3. Llama 70B Async Spec K=6 f=3 — Batch Size Sweep

### B=1 (ns=128)

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --backup jit --numseqs 128 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 270 tok/s |
| Avg draft step | 21.12 ms |

### B=1 (ns=8)

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --backup jit --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 238 tok/s |
| Avg draft step | 23.30 ms |

### B=2

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 2 --backup jit --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 409 tok/s |
| Avg draft step | 24.88 ms |

### B=4

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 4 --backup jit --numseqs 16 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 681 tok/s |
| Avg draft step | 28.96 ms |

### B=8

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 8 --backup jit --numseqs 32 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 1166 tok/s |
| Avg draft step | 35.29 ms |

### B=16 (uniform)

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 16 --backup jit --numseqs 64 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 1900 tok/s |
| Avg draft step | 43.84 ms |

### B=16 (non-uniform, f=1, flh=[1,1,1,1,1,1,1] flm=[4,2,1,0,0,0,0])

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 1 --b 16 --backup jit --numseqs 64 --output_len 512 --temp 0 --flh 1 1 1 1 1 1 1 --flm 4 2 1 0 0 0 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 2117 tok/s |

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
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --backup jit --numseqs 8 --output_len 512 --temp 0.7
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 229 tok/s |
| Avg draft step | 22.64 ms |

**Generations (temp=0.7, non-deterministic — sample only, will vary across runs):**

Prompt 1: Correctly computes 72 clips via step-by-step reasoning.
Prompt 2: Correctly computes $10.00 earnings.

### temp=1.0

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --backup jit --numseqs 8 --output_len 512 --temp 1.0
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
python -O bench/bench.py --size 32 --draft 0.6 --gpus 5 --spec --async --k 4 --f 2 --b 1 --backup jit --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 195 tok/s |
| Avg draft step | 17.93 ms |

**Qwen generations (temp=0) — TODO: capture full generations in next test run.**

---

## 6. Llama 70B Ablations (B=1, K=6 f=3)

### NO-JIT (default cudagraphs)

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --backup fast --numseqs 8 --output_len 512 --temp 0
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
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --backup jit --numseqs 8 --output_len 512 --temp 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 241 tok/s |
| Avg draft step | 22.89 ms |

### sampler_x=0.05, temp=1.3

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --backup jit --numseqs 8 --output_len 512 --temp 1.3 --x 0.05
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 128 tok/s |
| Avg draft step | 30.84 ms |

### sampler_x=0.25, temp=1.3

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --backup jit --numseqs 8 --output_len 512 --temp 1.3 --x 0.25
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 135 tok/s |
| Avg draft step | 30.59 ms |

### Custom fan_out_list: flh=[3,3,3,3,3,3,3] flm=[15,5,1,0,0,0,0]

**Command:**
```
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --backup jit --numseqs 8 --output_len 512 --temp 0 --flh 3 3 3 3 3 3 3 --flm 15 5 1 0 0 0 0
```

| Metric | Value |
|--------|-------|
| Decode Throughput | 224 tok/s |
| Avg draft step | 24.73 ms |

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
| **Llama 70B Async K=6 f=3 B=16** | 1900 | 5 GPUs, JIT |
| **Llama 70B Async K=6 f=1 B=16 nonunif** | 2117 | 5 GPUs, JIT |
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
