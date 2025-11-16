import os
import json
from datasets import load_dataset


def get_base_output_dir():
    """Get base output directory from HF_DATASETS_CACHE or default."""
    hf_cache = os.environ.get('HF_DATASETS_CACHE', '/tmp/hf_datasets_cache')
    print(f"Using HF_DATASETS_CACHE: {hf_cache}")
    return os.path.join(hf_cache, 'processed_datasets')


def download_gsm8k_data(num_samples=None):
    """Download GSM8K dataset samples and save as JSONL."""
    output_dir = os.path.join(get_base_output_dir(), "gsm8k")
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to 10k samples max
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    
    output_file = os.path.join(output_dir, f"gsm8k_data_{num_samples}.jsonl")

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file

    print(f"Loading GSM8K dataset...")
    
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
    except Exception as e:
        print(f"Error loading GSM8K dataset: {e}")
        raise

    total_samples = len(dataset)
    samples_to_process = min(num_samples, total_samples)
    
    print(f"Processing {samples_to_process} samples from {total_samples} total samples...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(samples_to_process):
            example = dataset[i]
            sample = {"text": example['question']}
            f.write(json.dumps(sample) + '\n')

            if i % 500 == 0:
                print(f"Processed {i}/{samples_to_process} samples...")

    print(f"Saved {samples_to_process} GSM8K samples to {output_file}")
    return output_file


def download_c4_data(num_samples=None):
    """Download C4 dataset samples and save as JSONL."""
    output_dir = os.path.join(get_base_output_dir(), "c4")
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to 10k samples max
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    
    output_file = os.path.join(output_dir, f"c4_data_{num_samples}.jsonl")

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file

    print(f"Loading C4 dataset...")
    
    try:
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading C4 dataset: {e}")
        raise

    print(f"Processing C4 samples (streaming)...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
                
            sample = {"text": example['text']}
            f.write(json.dumps(sample) + '\n')

            if i % 1000 == 0:
                print(f"Processed {i} samples...")

    samples_processed = min(i + 1, num_samples) if 'i' in locals() else 0
    print(f"Saved {samples_processed} C4 samples to {output_file}")
    return output_file


def download_ultrafeedback_data(num_samples=None):
    """Download UltraFeedback dataset samples and save as JSONL."""
    output_dir = os.path.join(get_base_output_dir(), "ultrafeedback")
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to 10k samples max
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    
    output_file = os.path.join(output_dir, f"ultrafeedback_data_{num_samples}.jsonl")

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file

    print(f"Loading UltraFeedback dataset...")
    
    try:
        dataset = load_dataset("openbmb/UltraFeedback", split="train")
    except Exception as e:
        print(f"Error loading UltraFeedback dataset: {e}")
        raise

    total_samples = len(dataset)
    samples_to_process = min(num_samples, total_samples)
    
    print(f"Processing {samples_to_process} samples from {total_samples} total samples...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(samples_to_process):
            example = dataset[i]
            # Use the instruction field as the main text
            sample = {"text": example['instruction']}
            f.write(json.dumps(sample) + '\n')

            if i % 500 == 0:
                print(f"Processed {i}/{samples_to_process} samples...")

    print(f"Saved {samples_to_process} UltraFeedback samples to {output_file}")
    return output_file


def download_humaneval_data(num_samples=None):
    """Download OpenAI HumanEval dataset samples and save as JSONL."""
    output_dir = os.path.join(get_base_output_dir(), "humaneval")
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to 10k samples max
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    
    output_file = os.path.join(output_dir, f"humaneval_data_{num_samples}.jsonl")

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file

    print(f"Loading HumanEval dataset...")
    
    try:
        dataset = load_dataset("openai/openai_humaneval", split="test")
    except Exception as e:
        print(f"Error loading HumanEval dataset: {e}")
        raise

    total_samples = len(dataset)
    samples_to_process = min(num_samples, total_samples)
    
    print(f"Processing {samples_to_process} samples from {total_samples} total samples...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(samples_to_process):
            example = dataset[i]
            # Use the prompt field as the main text
            sample = {"text": example['prompt']}
            f.write(json.dumps(sample) + '\n')

            if i % 100 == 0:
                print(f"Processed {i}/{samples_to_process} samples...")

    print(f"Saved {samples_to_process} HumanEval samples to {output_file}")
    return output_file


def download_alpaca_data(num_samples=None):
    """Download Alpaca dataset samples and save as JSONL."""
    output_dir = os.path.join(get_base_output_dir(), "alpaca")
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to 10k samples max
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    
    output_file = os.path.join(output_dir, f"alpaca_data_{num_samples}.jsonl")

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file

    print(f"Loading Alpaca dataset...")
    
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
    except Exception as e:
        print(f"Error loading Alpaca dataset: {e}")
        raise

    total_samples = len(dataset)
    samples_to_process = min(num_samples, total_samples)
    
    print(f"Processing {samples_to_process} samples from {total_samples} total samples...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(samples_to_process):
            example = dataset[i]
            # Combine instruction and input if available, otherwise just instruction
            text = example['instruction']
            if example.get('input', '').strip():
                text = f"{text}\n\n{example['input']}"
            
            sample = {"text": text}
            f.write(json.dumps(sample) + '\n')

            if i % 500 == 0:
                print(f"Processed {i}/{samples_to_process} samples...")

    print(f"Saved {samples_to_process} Alpaca samples to {output_file}")
    return output_file


def download_all_datasets(num_samples=None):
    """Download all datasets."""
    print("Downloading all datasets...")
    
    datasets = [
        ("GSM8K", download_gsm8k_data),
        ("C4", download_c4_data),
        ("UltraFeedback", download_ultrafeedback_data),
        ("HumanEval", download_humaneval_data),
        ("Alpaca", download_alpaca_data),
    ]
    
    output_files = {}
    for name, download_func in datasets:
        print(f"\n{'='*50}")
        print(f"Downloading {name}...")
        print('='*50)
        
        try:
            output_file = download_func(num_samples)
            output_files[name] = output_file
            print(f"✓ Successfully downloaded {name} to {output_file}")
        except Exception as e:
            print(f"✗ Failed to download {name}: {e}")
            output_files[name] = None
    
    return output_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face")
    parser.add_argument("--num-samples", type=int, default=None, 
                       help="Number of samples to download (default: all)")
    
    args = parser.parse_args()
    
    output_files = download_all_datasets(args.num_samples)
    
    print(f"\n{'='*60}")
    print("Download Summary:")
    print('='*60)
    for name, file_path in output_files.items():
        if file_path:
            print(f"✓ {name}: {file_path}")
        else:
            print(f"✗ {name}: Failed")