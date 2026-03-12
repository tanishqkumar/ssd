"""Paths for baseline (SGLang/vLLM) benchmarking.

All paths can be overridden via environment variables. Set these before running
the benchmark scripts, or edit the defaults below to match your setup.
"""
import os


def _required_env(var_name: str, note: str) -> str:
    value = os.environ.get(var_name)
    if value:
        return value
    raise RuntimeError(f"Missing required env var {var_name}. {note}")


HF_CACHE_DIR = _required_env(
    "SSD_HF_CACHE",
    "Set it to your HuggingFace cache hub directory (for example: /path/to/huggingface/hub).",
)

DATASET_DIR = _required_env(
    "SSD_DATASET_DIR",
    "Set it to your processed dataset directory (for example: /path/to/processed_datasets).",
)
DATASET_PATHS = {
    "humaneval":     f"{DATASET_DIR}/humaneval/humaneval_data_10000.jsonl",
    "alpaca":        f"{DATASET_DIR}/alpaca/alpaca_data_10000.jsonl",
    "c4":            f"{DATASET_DIR}/c4/c4_data_10000.jsonl",
    "gsm":           f"{DATASET_DIR}/gsm8k/gsm8k_data_10000.jsonl",
    "ultrafeedback": f"{DATASET_DIR}/ultrafeedback/ultrafeedback_data_10000.jsonl",
}

EAGLE3_SPECFORGE_70B = os.environ.get(
    "SSD_EAGLE3_SPECFORGE_70B",
    f"{HF_CACHE_DIR}/models--lmsys--SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge",
)
EAGLE3_YUHUILI_8B = os.environ.get(
    "SSD_EAGLE3_8B",
    f"{HF_CACHE_DIR}/models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B",
)
EAGLE3_QWEN_32B = os.environ.get(
    "SSD_EAGLE3_QWEN_32B",
    f"{HF_CACHE_DIR}/models--RedHatAI--Qwen3-32B-speculator.eagle3",
)

MODELS = {
    "llama_70b": os.environ.get(
        "BENCH_LLAMA_70B",
        f"{HF_CACHE_DIR}/models--meta-llama--Llama-3.3-70B-Instruct",
    ),
    "llama_1b": os.environ.get(
        "BENCH_LLAMA_1B",
        f"{HF_CACHE_DIR}/models--meta-llama--Llama-3.2-1B-Instruct",
    ),
    "qwen_32b": os.environ.get(
        "BENCH_QWEN_32B",
        f"{HF_CACHE_DIR}/models--Qwen--Qwen3-32B",
    ),
    "qwen_0.6b": os.environ.get(
        "BENCH_QWEN_06B",
        f"{HF_CACHE_DIR}/models--Qwen--Qwen3-0.6B",
    ),
    "eagle3_llama_70b": os.environ.get(
        "BENCH_EAGLE3_LLAMA_70B",
        "lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge",
    ),
    "eagle3_qwen_32b": os.environ.get(
        "BENCH_EAGLE3_QWEN_32B",
        "Zhihu-ai/Zhi-Create-Qwen3-32B-Eagle3",
    ),
}


def resolve_snapshot(path: str) -> str:
    """Resolve a HF cache directory to its snapshot containing config.json."""
    if os.path.exists(os.path.join(path, "config.json")):
        return path
    snapshots = os.path.join(path, "snapshots")
    if os.path.isdir(snapshots):
        for d in os.listdir(snapshots):
            candidate = os.path.join(snapshots, d)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
                return candidate
    if os.path.isdir(path):
        for d in os.listdir(path):
            candidate = os.path.join(path, d)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
                return candidate
    raise FileNotFoundError(f"No snapshot (config.json) found under {path}")
