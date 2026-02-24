# Speculative Speculative Decoding

A lightweight inference engine for research on async tree-based speculative decoding.

## Setup

Requires Python 3.11+ and CUDA GPUs (H100s recommended).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/tanishqkumar/ssd
cd ssd
uv sync
source .venv/bin/activate
python -c "from ssd import LLM; print('ok')"
```

### Paths

Before running anything, you need to tell SSD where your model weights and datasets live.
All paths are configured in `ssd/paths.py`. Open it and edit the defaults, or set the
corresponding environment variables:

- **`SSD_CUDA_ARCH`** — your GPU's compute capability. defaults to `9.0` (H100). set to
  `8.0` for A100, `8.9` for L40/4090, etc. this controls FlashInfer kernel compilation.
- **`SSD_HF_CACHE`** — directory where huggingface model snapshots are stored, i.e. the
  directory containing folders like `models--meta-llama--Llama-3.1-8B-Instruct/`. if you
  downloaded models with `huggingface-cli download`, this is your `HF_HOME/hub/` directory.
- **`SSD_DATASET_DIR`** — directory with preprocessed benchmark datasets (jsonl files).
  you can generate these with `python scripts/get_data_from_hf.py`.

You can also override the specific model snapshot paths (`SSD_TARGET_MODEL`, `SSD_DRAFT_MODEL`,
`SSD_EAGLE3_SPECFORGE`) if your snapshots live at non-standard locations.

## Quick Start

All benchmarks run from the `bench/` directory. The default dataset is GSM8K; use `--all` to
run across humaneval, alpaca, gsm, and ultrafeedback combined.

**Autoregressive baselines:**

```bash
# Llama 8B, 1 GPU
python bench.py --size 8 --numseqs 8 --output_len 128

# Llama 70B, 4 GPUs
python bench.py --size 70 --gpus 4 --numseqs 8 --output_len 512
```

**Synchronous speculative decoding** (target and draft take turns on the same GPUs):

```bash
# Llama 70B target + 1B draft, 4 GPUs, K=6 speculation depth
python bench.py --size 70 --draft 1 --gpus 4 --spec --k 6 --b 1 --numseqs 8 --output_len 512
```

**Async speculative decoding** (target and draft run concurrently on separate GPUs):

```bash
# Llama 70B target (4 GPUs) + 1B draft (1 GPU), K=6, F=3 fan-out
python bench.py --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --numseqs 8 --output_len 512

# same but batched (B=16) across all datasets
python bench.py --size 70 --draft 1 --gpus 5 --spec --async --k 7 --f 3 --b 16 --numseqs 128 --output_len 512 --all
```

**EAGLE3 async speculative decoding** (uses a small draft head conditioned on target activations):

```bash
# Llama 70B target + EAGLE3 draft head, 5 GPUs, K=7
python bench.py --size 70 --gpus 5 --eagle --async --k 7 --f 3 --b 1 --numseqs 8 --output_len 512

# with non-uniform fan-out schedule and batching
python bench.py --size 70 --gpus 5 --eagle --async --k 6 --b 4 --numseqs 16 --output_len 512 --flh 3 3 3 3 3 3 3 --flm 15 5 1 0 0 0 0
```

**Other options:**

```bash
--temp 0.7              # sampling temperature (default: 0, greedy)
--eager                 # disable CUDA graphs (slower but useful for debugging)
--backup jit            # JIT fallback on cache miss (default)
--wandb --group myexp   # log metrics to wandb
--example               # use canned prompts and print generations
--all                   # benchmark across all datasets (4x numseqs total)
```

Use `--qwen` for Qwen models (e.g. `--qwen --size 32 --draft 0.6`).
See `bench/bench.py` for the full list of arguments.
