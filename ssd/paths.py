import os

# cuda arch for flashinfer kernel compilation. set this to match your gpu:
# "9.0" for H100/H200, "8.0" for A100, "8.9" for L40/4090, etc.
CUDA_ARCH = os.environ.get("SSD_CUDA_ARCH", "9.0")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", CUDA_ARCH)

# root directory where huggingface model snapshots are stored. each model
# lives under this as models--org--name/snapshots/<hash>/. if you downloaded
# models with `huggingface-cli download`, this is your HF_HOME/hub directory.
HF_CACHE_DIR = os.environ.get("SSD_HF_CACHE", "/data/shared/huggingface/hub")

# default target and draft model snapshot paths. these are full paths to the
# snapshot directory containing config.json. override if your models live
# somewhere else or you want to use different model sizes.
DEFAULT_TARGET = os.environ.get(
    "SSD_TARGET_MODEL",
    f"{HF_CACHE_DIR}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
)
DEFAULT_DRAFT = os.environ.get(
    "SSD_DRAFT_MODEL",
    f"{HF_CACHE_DIR}/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6",
)

# eagle3 specforge draft for 70b target. this is the draft head trained by
# lmsys specifically for llama-3.3-70b; if you're running eagle3 with a
# different target you'll need to point this at the right checkpoint.
EAGLE3_SPECFORGE_70B = os.environ.get(
    "SSD_EAGLE3_SPECFORGE",
    f"{HF_CACHE_DIR}/models--lmsys--SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge",
)

# directory containing preprocessed benchmark datasets (jsonl files).
# each dataset is a subdirectory with a file like humaneval_data_10000.jsonl.
# you can generate these with scripts/get_data_from_hf.py.
DATASET_DIR = os.environ.get(
    "SSD_DATASET_DIR",
    "/data/tkumar/huggingface/processed_datasets",
)
DATASET_PATHS = {
    "humaneval":     f"{DATASET_DIR}/humaneval/humaneval_data_10000.jsonl",
    "alpaca":        f"{DATASET_DIR}/alpaca/alpaca_data_10000.jsonl",
    "c4":            f"{DATASET_DIR}/c4/c4_data_10000.jsonl",
    "gsm":           f"{DATASET_DIR}/gsm8k/gsm8k_data_10000.jsonl",
    "ultrafeedback": f"{DATASET_DIR}/ultrafeedback/ultrafeedback_data_10000.jsonl",
}
