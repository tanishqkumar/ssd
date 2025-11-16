import os
import sys
import argparse
import json
import torch
from random import seed
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(__file__))
from plot_helpers import (
    prepare_batch_data,
    get_model_logits,
    analyze_top_f_overlap,
    aggregate_results,
    print_results,
    plot_results,
)

def get_model_paths_and_tp(target_size, dsize=1):
    """Get model paths and tensor parallelism configuration based on target and draft sizes."""
    cache_dir = "/data/tkumar/huggingface/hub"

    # Target model based on size argument
    if target_size == 1 or target_size == 3:
        target_model_name = f"Llama-3.2-{target_size}B-Instruct"
    else:
        target_model_name = f"Llama-3.1-{target_size}B-Instruct"
    target_path = os.path.join(
        cache_dir, f"models--meta-llama--{target_model_name}", "snapshots")

    # Draft model based on size argument
    if dsize == 1 or dsize == 3:
        draft_model_name = f"Llama-3.2-{dsize}B-Instruct"
    else:
        draft_model_name = f"Llama-3.1-{dsize}B-Instruct"
    draft_path = os.path.join(
        cache_dir, f"models--meta-llama--{draft_model_name}", "snapshots")

    # Get actual snapshot directories
    target_snapshot_dirs = [d for d in os.listdir(
        target_path) if os.path.isdir(os.path.join(target_path, d))]
    if target_snapshot_dirs:
        target_path = os.path.join(target_path, target_snapshot_dirs[0])
    else:
        raise FileNotFoundError(
            f"No snapshot directory found in {target_path}")

    draft_snapshot_dirs = [d for d in os.listdir(
        draft_path) if os.path.isdir(os.path.join(draft_path, d))]
    if draft_snapshot_dirs:
        draft_path = os.path.join(draft_path, draft_snapshot_dirs[0])
    else:
        raise FileNotFoundError(f"No snapshot directory found in {draft_path}")

    return target_path, draft_path


def load_models(target_path, draft_path, device):
    """Load target and draft models without tensor parallelism."""
    print(f"Loading target model from: {target_path}")
    print(f"Loading draft model from: {draft_path}")

    target_model = AutoModelForCausalLM.from_pretrained(
        target_path, dtype=torch.bfloat16, device_map="auto")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_path, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(target_path)

    target_model.eval()
    draft_model.eval()

    return target_model, draft_model, tokenizer


def get_prompts(tokenizer, target_path, num_prompts, seq_len, c4=False):
    """Get prompts from target model."""
    if c4:
        print("Loading C4 dataset...")
        # Use C4 data from processed datasets directory
        c4_file_path = "/data/tkumar/huggingface//processed_datasets/c4/c4_data_5000.jsonl"
        if not os.path.exists(c4_file_path):
            print(f"C4 file not found at {c4_file_path}, defaulting to GSM8K")
            c4 = False
    
    if not c4:
        print("Loading GSM8K dataset...")
        # Default to GSM8K data from processed datasets
        gsm_file_path = "/data/tkumar/huggingface//processed_datasets/gsm8k/gsm8k_data_10000.jsonl"
        if not os.path.exists(gsm_file_path):
            raise FileNotFoundError(
                f"GSM8K file not found at {gsm_file_path}")

        prompts = []
        with open(gsm_file_path, 'r') as f:
            for i, line in enumerate(f):
                if len(prompts) >= num_prompts:  # Load num_prompts prompts
                    break

                data = json.loads(line.strip())
                text = data["text"]

                # Tokenize the text
                tokens = tokenizer.encode(text, add_special_tokens=False)

                # Cap to max(text field len in tokens, 256 tokens)
                target_len = max(len(tokens), 256)

                if len(tokens) >= target_len:
                    truncated_tokens = tokens[:target_len]
                else:
                    # If text is shorter than target_len, use all tokens
                    truncated_tokens = tokens

                prompts.append(truncated_tokens)
    else:
        prompts = []
        with open(c4_file_path, 'r') as f:
            for i, line in enumerate(f):
                if len(prompts) >= num_prompts:  # Load num_prompts prompts
                    break

                data = json.loads(line.strip())
                text = data["text"]

                # Tokenize the text
                tokens = tokenizer.encode(text, add_special_tokens=False)

                # Cap to max(text field len in tokens, 256 tokens)
                target_len = max(len(tokens), 256)

                if len(tokens) >= target_len:
                    truncated_tokens = tokens[:target_len]
                else:
                    # If text is shorter than target_len, use all tokens
                    truncated_tokens = tokens

                prompts.append(truncated_tokens)

    return prompts


def main():
    # Set random seed for reproducibility
    seed(42)
    torch.manual_seed(42)
    
    # Set device to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Llama 70B model with tp=4
    target_size = 70
    draft_size = 1  # Dummy value, not used
    
    # Get model paths
    target_path, draft_path = get_model_paths_and_tp(target_size, draft_size)
    
    # Load models and tokenizer
    print("Loading Llama 70B model...")
    target_model, draft_model, tokenizer = load_models(target_path, draft_path, device)

    # Load first prompt (defaults to GSM8K)
    prompts = get_prompts(tokenizer, target_path, 1, 256, c4=True)
    print(f'Loaded {len(prompts)} prompts')
    
    if len(prompts) > 0:
        first_prompt = prompts[0]
        print(f"First prompt tokens: {first_prompt}")
        print(f"First prompt text: {tokenizer.decode(first_prompt)}")
        
        # Prepare input tensor
        input_ids = torch.tensor([first_prompt], dtype=torch.long, device=device)
        
        # Run inference
        print("Running inference...")
        with torch.no_grad():
            outputs = target_model(input_ids)
            logits = outputs.logits
            
        print(f"Output logits shape: {logits.shape}")
        print(f"Sample logits (first 10 of last position): {logits[0, -1, :10]}")


if __name__ == "__main__":
    main()
