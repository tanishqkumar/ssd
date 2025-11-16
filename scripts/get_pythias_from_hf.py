#!/usr/bin/env python3
"""
Script to download Pythia models from Hugging Face Hub.
"""

import os
from huggingface_hub import snapshot_download


def download_pythia_models():
    """Download Pythia models to hub directory structure."""

    # Define models to download
    models = [
        "EleutherAI/pythia-12b",
        "EleutherAI/pythia-410m"
    ]

    # Set cache directory to the specified target location
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
    print(f"{cache_dir}/models--EleutherAI--pythia-<size>/snapshots/<hash>/")


if __name__ == "__main__":
    download_pythia_models()
