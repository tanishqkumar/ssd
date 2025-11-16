import os
import sys
import time
import argparse
import json
import pickle
import torch
import numpy as np
from random import randint, seed, random, shuffle
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset

# Import the verify function
sys.path.append(os.path.dirname(__file__))
from hf_verify import verify


def load_c4_prompts(cache_dir, model_path, num_prompts=1000, max_length=1024):
    """Load and tokenize C4 dataset snippets, with caching."""
    cache_file = os.path.join(
        cache_dir, f"c4_prompts_{num_prompts}_{max_length}.pkl")

    if os.path.exists(cache_file):
        print(f"Loading cached C4 prompts from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Loading C4 dataset and tokenizing {num_prompts} prompts...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load C4 dataset - use allenai/c4 instead of deprecated c4
    try:
        dataset = load_dataset(
            "allenai/c4", "en", split="train", streaming=True)
    except:
        raise ImportError(
            "datasets library not available. Please install with: pip install datasets")

    prompts = []
    with tqdm(total=num_prompts, desc="Processing C4 dataset") as pbar:
        for i, example in enumerate(dataset):
            if i >= num_prompts:
                break

            text = example['text']
            # Tokenize and truncate to max_length
            tokens = tokenizer.encode(
                text, max_length=max_length, truncation=True)

            # Skip very short prompts
            if len(tokens) >= 50:
                prompts.append(tokens)

            pbar.update(1)

    # Cache the tokenized prompts
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(prompts, f)

    print(f"Cached {len(prompts)} tokenized prompts to {cache_file}")
    return prompts


def get_model_paths():
    """Get model paths following bench.py pattern."""
    cache_dir = "/data/tkumar/huggingface/hub"

    # Target model (8B)
    target_model_name = "Llama-3.1-8B-Instruct"
    # target_model_name = "Llama-3.1-70B-Instruct"
    target_path = os.path.join(
        cache_dir, f"models--meta-llama--{target_model_name}", "snapshots")

    # Draft model (1B)
    draft_model_name = "Llama-3.2-1B-Instruct"
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


def greedy_decode_k_tokens(model, tokenizer, input_ids, attention_mask, k):
    """Greedily decode K tokens from the draft model."""
    model.eval()
    
    # Clone inputs to avoid modifying originals
    current_ids = input_ids.clone()
    current_mask = attention_mask.clone()
    
    generated_tokens = []
    
    with torch.no_grad():
        for step in range(k):
            # Forward pass
            outputs = model(current_ids, attention_mask=current_mask)
            logits = outputs.logits  # [B, seq_len, vocab_size]
            
            # Get next token (greedy)
            next_token_logits = logits[:, -1, :]  # [B, vocab_size]
            next_tokens = next_token_logits.argmax(dim=-1)  # [B]
            
            # Append to sequences
            current_ids = torch.cat([current_ids, next_tokens.unsqueeze(1)], dim=1)
            current_mask = torch.cat([current_mask, torch.ones(current_mask.size(0), 1, device=current_mask.device)], dim=1)
            
            # Store generated tokens
            generated_tokens.append(next_tokens)
    
    # Stack generated tokens: [K, B] -> [B, K]
    generated_tokens = torch.stack(generated_tokens, dim=0).T  # [B, K]
    
    return current_ids, current_mask, generated_tokens


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Speculative decoding verification with HF models')
    parser.add_argument('--K', type=int, default=4, help='Number of tokens to decode and verify (default: 4)')
    parser.add_argument('--temp', type=float, default=0.0, help='Target model temperature (default: 0.7)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
    args = parser.parse_args()
    
    K = args.K
    target_temp = args.temp
    batch_size = args.batch_size

    # Set device to CUDA H100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")

    # Set random seed for reproducibility
    seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Get model paths
    target_path, draft_path = get_model_paths()
    cache_dir = "/data/tkumar/huggingface/hub"

    print(f"Loading target model from: {target_path}")
    print(f"Loading draft model from: {draft_path}")

    # Load models and tokenizer
    print("Loading models...")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_path, dtype=torch.bfloat16, device_map="auto")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_path, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(target_path)

    target_model.eval()
    draft_model.eval()

    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    gsm8k_file_path = "/data/tkumar/huggingface/gsm8k/gsm8k_data_all.jsonl"
    if not os.path.exists(gsm8k_file_path):
        raise FileNotFoundError(f"GSM8K file not found at {gsm8k_file_path}")

    prompts = []
    with open(gsm8k_file_path, 'r') as f:
        for i, line in enumerate(f):
            if len(prompts) >= 100:  # Load 100 prompts
                break

            data = json.loads(line.strip())
            text = data["text"]

            # Tokenize the text
            tokens = tokenizer.encode(text, add_special_tokens=False)

            # Use prompts of varying lengths (don't pad to fixed length)
            # Cap at reasonable length to avoid memory issues
            max_prompt_len = 256
            if len(tokens) > max_prompt_len:
                tokens = tokens[:max_prompt_len]

            prompts.append(tokens)

    print(f"Loaded {len(prompts)} prompts")
    
    # Process in batches
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    total_accepted = 0
    total_sequences = 0
    
    print(f"\nRunning speculative decoding verification with K={K}, target_temp={target_temp}")
    print("Draft model uses greedy decoding (temp=0)")
    
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    with tqdm(total=num_batches, desc="Processing batches") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]
            current_batch_size = len(batch_prompts)
            
            # Find max length in this batch
            max_len = max(len(prompt) for prompt in batch_prompts)
            
            # Pad prompts to same length within batch
            padded_prompts = []
            for prompt in batch_prompts:
                if len(prompt) < max_len:
                    padded_prompt = prompt + [pad_token] * (max_len - len(prompt))
                else:
                    padded_prompt = prompt
                padded_prompts.append(padded_prompt)
            
            # Convert to tensor
            input_ids = torch.tensor(padded_prompts, dtype=torch.long, device=device)
            attention_mask = (input_ids != pad_token).long()
            
            # Step 1: Greedy decode K tokens with draft model
            extended_ids, extended_mask, speculations = greedy_decode_k_tokens(
                draft_model, tokenizer, input_ids, attention_mask, K)
            
            # Step 2: Pass extended sequences through target model to get logits at speculation positions
            with torch.no_grad():
                target_outputs = target_model(extended_ids, attention_mask=extended_mask)
                target_logits_full = target_outputs.logits  # [B, seq_len + K, vocab_size]
                
                # IMPORTANT: Get target logits for positions where draft tokens were generated
                # If draft generated tokens at positions L, L+1, ..., L+K-1,
                # then target logits should be at positions L-1, L, ..., L+K-2
                original_seq_len = input_ids.size(1)
                target_logits = target_logits_full[:, original_seq_len-1:original_seq_len-1+K, :]  # [B, K, vocab_size]
                
                # For draft logits, we need to get them from the same positions
                draft_outputs = draft_model(extended_ids, attention_mask=extended_mask)
                draft_logits_full = draft_outputs.logits  # [B, seq_len + K, vocab_size]
                draft_logits = draft_logits_full[:, original_seq_len-1:original_seq_len-1+K, :]  # [B, K, vocab_size]
            
            # Step 3: Set up temperatures
            temperatures_target = torch.full((current_batch_size,), target_temp, device=device)
            temperatures_draft = torch.zeros((current_batch_size,), device=device)  # Draft is greedy
            
            # Step 4: Run verification
            accepted_suffixes, recovery_tokens = verify(
                target_logits,
                draft_logits,
                speculations,
                temperatures_target,
                temperatures_draft
            )
            
            # Step 5: Count accepted tokens
            batch_accepted = sum(len(suffix) for suffix in accepted_suffixes)
            total_accepted += batch_accepted
            total_sequences += current_batch_size
            
            pbar.update(1)
    
    # Print results
    avg_accepted = total_accepted / total_sequences
    print(f"\n" + "="*60)
    print("SPECULATIVE DECODING VERIFICATION RESULTS")
    print("="*60)
    print(f"Total sequences processed: {total_sequences}")
    print(f"K (speculation length): {K}")
    print(f"Target model temperature: {target_temp}")
    print(f"Draft model temperature: 0.0 (greedy)")
    print(f"Total tokens accepted: {total_accepted}")
    print(f"Average accepted tokens per sequence: {avg_accepted:.3f}")
    print(f"Acceptance rate: {avg_accepted/K:.3f} ({avg_accepted/K*100:.1f}%)")


if __name__ == "__main__":
    main()
