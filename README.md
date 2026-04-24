<h1 align="center">Speculative Speculative Decoding</h1>

<h3 align="center">
  <a href="https://arxiv.org/pdf/2603.03251">Paper</a>
</h3>

<p align="center">
  <img width="800"
       src="assets/ssd fig1 readme.png" />
</p>

> *"In all fictions, each time a man meets diverse alternatives, he chooses one and eliminates the others; in the work of the almost unfathomable Ts'ui Pên, he chooses — simultaneously — all of them."*
>
> — Jorge Luis Borges, "The Garden of Forking Paths" (1941)

**SSD is a new LLM inference algorithm. It is exact, and it is extremely fast.**

SSD is a new type of speculative decoding (SD). In normal SD, a small and fast model guesses the next few tokens that a larger slower model may generate, and the large model then verifies them in one forward pass: drafting and verification happen one after the other on the same hardware.

In SSD, they happen in parallel, on distinct hardware. The small model anticipates likely verification outcomes in advance, and speculates for all of them at once. If it guessed correctly, the speculation can be returned immediately so drafting overhead is eliminated entirely.

This custom inference engine supports:
- A reference implementation of the SSD algorithm
- Optimized SD and autoregressive baselines
- Qwen3 + Llama3 model families
- Tensor Parallelism
- PagedAttention, CUDAgraphs, torch compilation, prefix caching

## Setup

Requirements: Python 3.11+, CUDA >= 12.8. This code was written and tested on H100s. 

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# if `uv` is not found in this shell:
export PATH="$HOME/.local/bin:$PATH"
```

Then: 

```bash
git clone https://github.com/tanishqkumar/ssd && cd ssd
uv sync                    # core SSD deps
# uv sync --extra scripts  # add deps used by scripts/
source .venv/bin/activate
python -c "from ssd import LLM; print('ok')"
```

Set paths via environment variables. `SSD_HF_CACHE` should point to the HuggingFace **hub** directory — this is the directory that contains `models--org--name/` subdirectories (e.g. `/data/huggingface/hub`, not `/data/huggingface/`). `SSD_DATASET_DIR` should point to the directory containing the dataset subdirectories (`humaneval/`, `alpaca/`, etc).

```bash
export SSD_HF_CACHE=/path/to/huggingface/hub
export SSD_DATASET_DIR=/path/to/processed_datasets
export SSD_CUDA_ARCH=9.0   # 9.0=H100, 8.0=A100, 8.9=L40/4090
```

### Download models + datasets

If you already have the models downloaded via `huggingface-cli` or similar, you can skip straight to datasets — just make sure `SSD_HF_CACHE` points to the right place. The download scripts require the `scripts` extra: `uv sync --extra scripts`.

```bash
# models (uses SSD_HF_CACHE)
python scripts/download_from_hf.py llama

# datasets (writes to $HF_DATASETS_CACHE/processed_datasets)
export HF_DATASETS_CACHE=/path/to  # parent of SSD_DATASET_DIR
python scripts/get_data_from_hf.py --num-samples 10000
```

## Setup (AMD MI300x / ROCm)

Runs on AMD Instinct MI300x with ROCm 7.2, PyTorch 2.9.1, and
[FlashInfer v0.5.3+amd.2](https://github.com/ROCm/flashinfer/releases/tag/v0.5.3%2Bamd.2)
(ships [PR #214](https://github.com/ROCm/flashinfer/pull/214) kernel fixes).

**1. Build the Docker image** (from the FlashInfer release source):

```bash
git clone https://github.com/tanishqkumar/ssd /home/yourname/ssd && cd /home/yourname/ssd

git clone --branch v0.5.3+amd.2 --depth 1 \
    https://github.com/ROCm/flashinfer.git /home/yourname/tmp/flashinfer-build
cd /home/yourname/tmp/flashinfer-build
docker build \
    --build-arg ROCM_VERSION=7.2 \
    --build-arg PY_VERSION=3.12 \
    --build-arg TORCH_VERSION=2.9.1 \
    -t flashinfer-0.5.3.amd2_rocm7.2 \
    -f .devcontainer/rocm/Dockerfile .
```

**2. Start and enter the container**:

```bash
docker run -dit \
  --name ssd \
  --privileged --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size 64G \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /home:/home \
  flashinfer-0.5.3.amd2_rocm7.2 \
  /bin/bash
docker exec -u 0 -it ssd bash
```

**3. Activate the env and install SSD + flash-attn**:

```bash
export MAMBA_EXE=/bin/micromamba
export MAMBA_ROOT_PREFIX=/opt/conda
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate flashinfer-py3.12-torch2.9.1-rocm7.2

pip install --no-build-isolation -ve /home/yourname/tmp/flashinfer-build
python -c "import torch; print(torch.__version__)"       # should print 2.9.1+...
python -c "import flashinfer; print(flashinfer.__version__)"  # should print a version string

cd /home/yourname/ssd
bash setup_rocm.sh
```

`setup_rocm.sh` runs `pip install -e .` and builds `flash-attn` from
upstream commit `0f82fea` (overridable via `FLASH_ATTN_REF`) with the CK
backend for `gfx942`.

Tree-decode backend is selectable via `SSD_TREE_DECODE_BACKEND={flashinfer,sdpa}`
(default: `flashinfer`).

**4: Download models and datasets**

```bash
# Set environment variables
export SSD_HF_CACHE=/path/to/huggingface/hub    # e.g. /home/<your-username>/hf_cache
export SSD_DATASET_DIR=$SSD_HF_CACHE/processed_datasets
export HSA_NO_SCRATCH_RECLAIM=1

# Login to HuggingFace (needed for gated models like Llama)
pip install huggingface_hub datasets
huggingface-cli login

# Download models
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --cache-dir $SSD_HF_CACHE
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --cache-dir $SSD_HF_CACHE

# For 70B benchmarks (requires ~140GB disk):
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --cache-dir $SSD_HF_CACHE

# Download and preprocess benchmark datasets
HF_DATASETS_CACHE=$SSD_HF_CACHE python scripts/get_data_from_hf.py
```

The dataset script downloads HumanEval, Alpaca, C4, GSM8K, and UltraFeedback, then saves processed JSONL files to `$SSD_DATASET_DIR`.

## Usage

All commands below run from inside the `bench/` directory. Large models (Llama-3 70B, Qwen-3 32B) take a few minutes for load/warmup/compile before generation starts. Always use `python -O` to disable debug overhead.

### Benchmarks

Use `--all` for full eval across four datasets. Since different data distributions are predictable to varying degrees, the speed of SD/SSD depends a lot on the dataset. Averaging over many prompts from many types of datasets 
gives an overall picture. `--numseqs` is per-dataset, so `--numseqs 128 --all` runs 128 × 4 = 512 prompts total.

```bash
cd bench

# AR — Llama 70B, 4 GPUs
python -O bench.py --llama --size 70 --gpus 4 --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# Sync spec decode — 70B target + 1B draft, 4 GPUs, k=6
python -O bench.py --llama --size 70 --gpus 4 --spec --k 6 --b 1 --temp 0 --numseqs 128 --output_len 512 --all

# Async spec decode (SSD) — 70B target (4 GPUs) + 1B draft (1 GPU), k=7, f=3
python -O bench.py --llama --size 70 --gpus 5 --spec --async --k 7 --f 3 --b 1 --temp 0 --numseqs 128 --output_len 512 --all
```

Use `--qwen --size 32` for Qwen models. See `bench/bench.py` for full args. For SGLang/vLLM baselines, see `bench/README.md`.

### Chat

Interactive streaming chat with Llama-3.1 70B only. Supports AR, sync SD, and async SD (SSD). Pass `--metrics` to print token count, speed, and TTFT after each response.

```bash
cd bench

# AR — 4 GPUs
python -O chat.py --ssd --gpus 4

# Sync spec decode — 4 GPUs, k=6
python -O chat.py --ssd --spec --k 6 --gpus 4

# Async spec decode (SSD) — 5 GPUs, k=7, f=3
python -O chat.py --ssd --spec --async --k 7 --f 3 --gpus 5 --metrics
```

SGLang and vLLM chat backends are also supported (launches their servers automatically) for comparison:

```bash
python -O chat.py --sglang        # spec decode
python -O chat.py --sglang --ar   # autoregressive
python -O chat.py --vllm          # spec decode
```

### Roadmap

Features that will be supported in the near future: 
- Draft data parallel (increase speculation cache size) on up to 4 devices to avoid getting compute bound
- OpenAI-compatible inference over HTTP
- New models and MoE support: GPT-OSS and Kimi-K2.5.

Contributions welcome! 

## Citation

Speculative Speculative Decoding will appear at ICLR 2026.

```bibtex
@misc{kumar2026speculativespeculativedecoding,
      title={Speculative Speculative Decoding},
      author={Tanishq Kumar and Tri Dao and Avner May},
      year={2026},
      eprint={2603.03251},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.03251},
}
```

## History

[![Star History Chart](https://api.star-history.com/svg?repos=tanishqkumar/ssd&type=Date)](https://star-history.com/#tanishqkumar/ssd&Date)
