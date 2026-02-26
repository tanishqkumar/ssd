#!/usr/bin/env python3
from huggingface_hub import snapshot_download


def download_llama_models():
    models = [
        # "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        # "meta-llama/Llama-3.1-70B-Instruct",
        # "meta-llama/Llama-3.2-1B-Instruct",
        # "meta-llama/Llama-3.2-3B-Instruct"
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
    print(f"{cache_dir}/models--meta-llama--Llama-<version>-<size>/snapshots/<hash>/")


if __name__ == "__main__":
    download_llama_models()
