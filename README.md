# Speculative Speculative Decoding

A lightweight inference engine for research on async tree-based speculative decoding.

## Installation

Requires Python 3.11+ and a CUDA GPU.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/tanishqkumar/ssd
cd ssd
uv sync

# Activate the environment
source .venv/bin/activate

# Verify installation
python -c "from ssd import LLM; print('ok')"
```

## Quick Start

To see a demo and benchmark, go to `cd bench/` directory and run: 

`python bench.py --size 8 --numseqs 4 --llama` (autoregressive, single-GPU by default)

`python bench.py --size 70 --gpus 4 --numseqs 4 --llama` (autoregressive, big model needs 4 GPUs)

`python bench.py --size 70 --draft 1 --gpus 4 --numseqs 4 --llama --spec --k 4` (sync spec, big model)

`python bench.py --size 70 --draft 1 --gpus 5 --numseqs 4 --llama --spec --async --k 4` (async spec, needs an extra GPU)

See `bench/bench.py` for all hyperparams (Saguaro sampling, batch size, temperature, etc). 

