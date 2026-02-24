#!/usr/bin/env python3
from huggingface_hub import snapshot_download


def download_qwen_models():
    models = [
        "Qwen/Qwen3-0.6B", 
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B", 
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B"
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
    print(f"{cache_dir}/models--Qwen--Qwen3-<size>/snapshots/<hash>/")

if __name__ == "__main__":
    download_qwen_models()
