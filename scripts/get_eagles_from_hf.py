#!/usr/bin/env python3
from huggingface_hub import snapshot_download


def download_eagle_models():
    models = [
        "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
        "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "RedHatAI/Qwen3-32B-speculator.eagle3"
    ]

    cache_dir = "/data/tkumar/huggingface/hub"

    for model_name in models:
        print(f"Downloading {model_name}...")
        try:
            snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                resume_download=True,
                local_files_only=False
            )
            print(f"✓ Successfully downloaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")

    print("\nDownload complete! Models are stored in:")
    print(f"{cache_dir}/models--<org>--<model>/snapshots/<hash>/")


if __name__ == "__main__":
    download_eagle_models()
