#!/bin/bash
# setup_rocm.sh - Set up ssd for AMD ROCm MI300x
# Build Docker from: https://github.com/ROCm/flashinfer/releases/tag/v0.5.3%2Bamd.2
# See README.md for Docker build instructions.
#
# Prerequisites: activate the micromamba environment before running this script:
#   export MAMBA_EXE=/bin/micromamba
#   export MAMBA_ROOT_PREFIX=/opt/conda
#   eval "$($MAMBA_EXE shell hook --shell bash)"
#   micromamba activate flashinfer-py3.12-torch2.9.1-rocm7.2
set -e

echo "=== ssd ROCm Setup ==="

# Verify the environment is activated
if ! python3 -c "import torch" 2>/dev/null; then
    echo "ERROR: torch not found. Activate the micromamba environment first:"
    echo "  export MAMBA_EXE=/bin/micromamba"
    echo "  export MAMBA_ROOT_PREFIX=/opt/conda"
    echo '  eval "$($MAMBA_EXE shell hook --shell bash)"'
    echo "  micromamba activate flashinfer-py3.12-torch2.9.1-rocm7.2"
    exit 1
fi

SSD_DIR="$(dirname "$(readlink -f "$0")")"
cd "$SSD_DIR"

echo "[1/6] Verifying GPU packages..."
python3 -c "import torch; print(f'  PyTorch {torch.__version__} (HIP: {torch.version.hip})')"
python3 -c "import triton; print(f'  Triton {triton.__version__}')"
echo "  GPUs: $(python3 -c 'import torch; print(torch.cuda.device_count())')"

echo "[2/6] Installing FlashInfer (if not already installed)..."
if python3 -c "import flashinfer" 2>/dev/null; then
    python3 -c "import flashinfer; print(f'  FlashInfer {flashinfer.__version__} already installed')"
else
    echo "  FlashInfer not found, installing from source..."
    pip install --no-build-isolation -e /workspace 2>&1 | tail -5
    python3 -c "import flashinfer; print(f'  FlashInfer {flashinfer.__version__} installed')"
fi

echo "[3/6] Installing ssd and dependencies..."
cd "$SSD_DIR"
rm -f uv.lock
pip install -e . 2>&1 | tail -3

echo "[4/6] Building flash-attn from source (CK backend for ROCm)..."
echo "  This is REQUIRED for CUDA/HIP graph mode. Expect 10-30 minutes."
echo "  (pip install flash-attn does NOT work on ROCm)"
if [ -z "$CUDA_HOME" ] && [ -d "/opt/rocm" ]; then
    export CUDA_HOME=/opt/rocm
    echo "  CUDA_HOME not set; defaulting to $CUDA_HOME for ROCm build"
fi
FLASH_ATTN_DIR="/tmp/flash-attention-build"
FLASH_ATTN_REF="${FLASH_ATTN_REF:-0f82fea}"
if python3 - <<'PY'
import pathlib

try:
    import flash_attn
    from flash_attn import flash_attn_with_kvcache, flash_attn_interface
except Exception as exc:
    print(f"  flash-attn compatibility check failed: {exc}")
    raise SystemExit(1)

interface_text = pathlib.Path(flash_attn_interface.__file__).read_text()
if "num_splits" in interface_text:
    print("  Installed flash-attn wrapper is incompatible with the ROCm extension; rebuilding")
    raise SystemExit(1)

print(f"  flash-attn {getattr(flash_attn, '__version__', 'unknown')} already installed and compatible")
PY
then
    :
else
    echo "  Using pinned flash-attention commit: $FLASH_ATTN_REF"
    if [ ! -d "$FLASH_ATTN_DIR/.git" ]; then
        rm -rf "$FLASH_ATTN_DIR"
        git clone https://github.com/Dao-AILab/flash-attention.git "$FLASH_ATTN_DIR"
    fi
    cd "$FLASH_ATTN_DIR"
    git -c safe.directory="$FLASH_ATTN_DIR" fetch origin --tags
    git -c safe.directory="$FLASH_ATTN_DIR" checkout "$FLASH_ATTN_REF"
    git -c safe.directory="$FLASH_ATTN_DIR" submodule update --init csrc/composable_kernel
    git -c safe.directory="$FLASH_ATTN_DIR" submodule update --init csrc/cutlass

    echo "  Removing incompatible flash-attn install (if present)..."
    python3 -m pip uninstall -y flash-attn >/dev/null 2>&1 || true
    python3 - <<'PY'
import pathlib
import shutil
import site

patterns = ("flash_attn", "flash_attn-*.dist-info", "flash_attn_2_cuda*.so")
paths = set(site.getsitepackages())
try:
    paths.add(site.getusersitepackages())
except Exception:
    pass

for base in sorted(pathlib.Path(path) for path in paths):
    if not base.exists():
        continue
    for pattern in patterns:
        for target in base.glob(pattern):
            print(f"    removing {target}")
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
PY

    rm -rf build dist flash_attn.egg-info
    pip install packaging wheel ninja psutil einops --quiet
    GPU_ARCHS="gfx942" pip install . --no-build-isolation --no-cache-dir 2>&1 | tee /tmp/flash-attn-build.log | tail -20
    cd "$SSD_DIR"
    if python3 - <<'PY'
import pathlib

from flash_attn import flash_attn_with_kvcache, flash_attn_interface

interface_text = pathlib.Path(flash_attn_interface.__file__).read_text()
if "num_splits" in interface_text:
    raise SystemExit(1)

print("  flash-attn built and installed successfully")
PY
    then
        :
    else
        echo ""
        echo "  *** WARNING: flash-attn build failed! ***"
        echo "  Graph mode will NOT work. You must use --eager flag."
        echo "  To build manually:"
        echo "    git clone https://github.com/Dao-AILab/flash-attention.git"
        echo "    cd flash-attention && git checkout $FLASH_ATTN_REF"
        echo "    GPU_ARCHS=gfx942 pip install . --no-build-isolation --no-cache-dir"
        echo ""
    fi
fi

echo "[5/6] Setting recommended environment variables..."
export TORCH_CUDA_ARCH_LIST=gfx942
export HSA_NO_SCRATCH_RECLAIM=1
echo "  TORCH_CUDA_ARCH_LIST=gfx942"
echo "  HSA_NO_SCRATCH_RECLAIM=1"

echo "[6/6] Running import test..."
export SSD_HF_CACHE="${SSD_HF_CACHE:-/tmp}"
export SSD_DATASET_DIR="${SSD_DATASET_DIR:-/tmp}"
python3 -c "
import ssd.paths
print('  ssd.paths OK (arch:', ssd.paths.CUDA_ARCH, ')')
from ssd.layers.attention import flash_attn_varlen_func
print('  flash_attn OK (module:', flash_attn_varlen_func.__module__, ')')
try:
    from flash_attn import flash_attn_with_kvcache
    print('  flash-attn CK backend: INSTALLED (graph mode OK)')
except ImportError:
    print('  flash-attn CK backend: NOT INSTALLED (eager mode only!)')
import flashinfer, torch
w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    torch.zeros(128*1024*1024, dtype=torch.uint8, device='cuda:0'), 'NHD', backend='fa2')
cu = torch.arange(3, dtype=torch.int32, device='cuda:0') * 4
ki = torch.tensor([0, 1, 2], dtype=torch.int32, device='cuda:0')
kx = torch.tensor([0, 1], dtype=torch.int32, device='cuda:0')
kl = torch.tensor([64, 64], dtype=torch.int32, device='cuda:0')
cm = torch.ones(2*4*128, dtype=torch.bool, device='cuda:0')
w.plan(cu, ki, kx, kl, 8, 2, 128, 64, custom_mask=cm, q_data_type=torch.bfloat16)
q = torch.randn(8, 8, 128, dtype=torch.bfloat16, device='cuda:0')
kc = torch.randn(2, 64, 2, 128, dtype=torch.bfloat16, device='cuda:0')
vc = torch.randn(2, 64, 2, 128, dtype=torch.bfloat16, device='cuda:0')
o = w.run(q, (kc, vc))
print(f'  FlashInfer plan()+run() OK (output: {o.shape})')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print('  All checks passed!')
"

echo ""
echo "=== Setup Complete ==="
echo "You still need to set these before running:"
echo "  export SSD_HF_CACHE=/path/to/huggingface/hub"
echo "  export SSD_DATASET_DIR=/path/to/processed_datasets"
echo "  export HSA_NO_SCRATCH_RECLAIM=1"
echo ""
echo "Run benchmark:"
echo "  python -O bench/bench.py --llama --size 8 --gpus 2 --spec --async --k 7 --f 3 --b 1 --temp 0 --numseqs 16 --output_len 128 --random"
