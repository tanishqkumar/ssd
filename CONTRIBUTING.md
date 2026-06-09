# Contributing to ssd

Thanks for your interest in contributing.

## Reporting Issues

Use [GitHub Issues](../../issues) to report bugs or request features. Include a clear description, reproduction steps, and your environment:

- OS and Python version
- CUDA version and GPU type
- PyTorch, Triton, and Transformers versions
- Model family, model size, GPU count, and command line
- Relevant environment variables such as `SSD_HF_CACHE`, `SSD_DATASET_DIR`, and `SSD_CUDA_ARCH`

For correctness or performance issues, include the benchmark or chat command, expected behavior, observed behavior, and any relevant logs or generated outputs.

## Development Setup

Requirements: Python 3.11+ and CUDA >= 12.8. The project README notes that this code was written and tested on H100s.

Install dependencies with `uv`:

```bash
uv sync
source .venv/bin/activate
```

Install the scripts extra when working with model or dataset download helpers:

```bash
uv sync --extra scripts
```

Set the expected local paths before running benchmarks or data scripts:

```bash
export SSD_HF_CACHE=/path/to/huggingface/hub
export SSD_DATASET_DIR=/path/to/processed_datasets
export SSD_CUDA_ARCH=9.0
```

Then verify that the package imports:

```bash
python -c "from ssd import LLM; print('ok')"
```

## Testing and Validation

Run the relevant import, unit, or benchmark checks before opening a pull request. At minimum, verify the package imports in the synced environment:

```bash
python -c "from ssd import LLM; print('ok')"
```

For changes to inference, scheduling, model execution, CUDA behavior, or benchmark logic, also run the smallest relevant benchmark or chat path from `bench/`:

```bash
cd bench
python -O bench.py --llama --size 70 --gpus 4 --b 1 --temp 0 --numseqs 1 --output_len 32
```

Adjust model family, size, GPU count, and SSD/speculative decoding flags to match the code path you changed. Add or update tests and documentation when behavior, commands, configuration, or expected results change.

## Pull Request Workflow

1. Fork the repository and create a branch from `main`:

   ```bash
   git checkout -b feat/short-description
   ```

2. Make your change. Add tests and update docs if behavior changes.
3. Run the relevant checks.
4. Open a PR against `main`. Describe what changed and why; link any related issue.

By opening a PR, you agree your contribution is licensed under the terms in [LICENSE](LICENSE).

## External Contributors

This repo is part of the AMD-AGI org. Non-AMD contributors need admin approval before being added as collaborators and must follow AMD's [open-source contribution guidelines](#).

For security issues, do **not** open a public issue. See [SECURITY.md](SECURITY.md) for private reporting options.
