#!/usr/bin/env python3
import argparse
from huggingface_hub import snapshot_download

MODELS = {
    "llama": [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ],
    "qwen": [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
    ],
    "eagle": [
        "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
        "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "RedHatAI/Qwen3-32B-speculator.eagle3",
    ],
}

CACHE_DIR = "/data/tkumar/huggingface/hub"


def download(models):
    for name in models:
        print(f"Downloading {name}...")
        try:
            snapshot_download(repo_id=name, cache_dir=CACHE_DIR, resume_download=True)
            print(f"  done")
        except Exception as e:
            print(f"  failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=[*MODELS.keys(), "all"])
    args = parser.parse_args()

    if args.group == "all":
        targets = [m for group in MODELS.values() for m in group]
    else:
        targets = MODELS[args.group]

    download(targets)
