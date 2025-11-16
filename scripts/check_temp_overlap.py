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

def load_c4_prompts(cache_dir, model_path, num_prompts=1000, max_length=1024):
    """Load and tokenize C4 dataset snippets, with caching."""
    cache_file = os.path.join(cache_dir, f"c4_prompts_{num_prompts}_{max_length}.pkl")
    
    if os.path.exists(cache_file):
        print(f"Loading cached C4 prompts from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Loading C4 dataset and tokenizing {num_prompts} prompts...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load C4 dataset - use allenai/c4 instead of deprecated c4
    try:
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    except:
        raise ImportError("datasets library not available. Please install with: pip install datasets")
        
    
    prompts = []
    with tqdm(total=num_prompts, desc="Processing C4 dataset") as pbar:
        for i, example in enumerate(dataset):
            if i >= num_prompts:
                break
                
            text = example['text']
            # Tokenize and truncate to max_length
            tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
            
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
    # target_model_name = "Llama-3.1-8B-Instruct"
    target_model_name = "Llama-3.1-70B-Instruct"
    target_path = os.path.join(cache_dir, f"models--meta-llama--{target_model_name}", "snapshots")
    
    # Draft model (1B)  
    draft_model_name = "Llama-3.2-1B-Instruct"
    draft_path = os.path.join(cache_dir, f"models--meta-llama--{draft_model_name}", "snapshots")
    
    # Get actual snapshot directories
    target_snapshot_dirs = [d for d in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, d))]
    if target_snapshot_dirs:
        target_path = os.path.join(target_path, target_snapshot_dirs[0])
    else:
        raise FileNotFoundError(f"No snapshot directory found in {target_path}")
        
    draft_snapshot_dirs = [d for d in os.listdir(draft_path) if os.path.isdir(os.path.join(draft_path, d))]
    if draft_snapshot_dirs:
        draft_path = os.path.join(draft_path, draft_snapshot_dirs[0])
    else:
        raise FileNotFoundError(f"No snapshot directory found in {draft_path}")
    
    return target_path, draft_path

def main():
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
    target_model = AutoModelForCausalLM.from_pretrained(target_path, torch_dtype=torch.bfloat16, tp_plan="auto")
    draft_model = AutoModelForCausalLM.from_pretrained(draft_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    
    # Ensure models are on CUDA
    target_model = target_model.to(device)
    draft_model = draft_model.to(device)
    
    # Load C4 dataset
    print("Loading C4 dataset...")
    c4_prompts = load_c4_prompts(cache_dir, target_path, num_prompts=100, max_length=256)
    
    # Sample batch of sequences
    batch_size = 32
    seq_len = 512
    batch_prompts = []
    
    for i in range(batch_size):
        prompt = c4_prompts[i % len(c4_prompts)]
        # Pad or truncate to exactly seq_len
        if len(prompt) >= seq_len:
            prompt = prompt[:seq_len]
        else:
            # Pad with tokenizer.pad_token_id or 0 if not available
            pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            prompt = prompt + [pad_token] * (seq_len - len(prompt))
        batch_prompts.append(prompt)
    
    # Convert to tensor and move to CUDA
    input_ids = torch.tensor(batch_prompts, device=device)
    print(f"Input batch shape: {input_ids.shape}")
    print(f"Input tensor device: {input_ids.device}")
    
    # Forward pass through both models
    print("Running forward passes...")
    with torch.no_grad():
        target_outputs = target_model(input_ids)
        draft_outputs = draft_model(input_ids)
    
    target_logits = target_outputs.logits  # [batch_size, seq_len, vocab_size]
    draft_logits = draft_outputs.logits    # [batch_size, seq_len, vocab_size]
    
    print(f"Target logits shape: {target_logits.shape}")
    print(f"Draft logits shape: {draft_logits.shape}")
    print(f"Target logits device: {target_logits.device}")
    print(f"Draft logits device: {draft_logits.device}")
    
    # Temperature and F values to sweep
    temperatures = [0.1, 0.3, 0.5, 0.7, 1.0]
    f_values = [1, 2, 4, 8]
    
    # For each position except the last (since we need next token)
    results = {}
    
    # Add progress bar for temperature and F sweeps
    total_combinations = len(temperatures) * len(f_values)
    with tqdm(total=total_combinations, desc="Analyzing temp/F combinations") as pbar:
        for temp in temperatures:
            for f in f_values:
                total_disagreements = 0
                overlap_count = 0
                
                # Sample from target model at each position
                for pos in range(seq_len - 1):  # -1 because we predict next token
                    # Get logits at this position for all sequences in batch
                    target_pos_logits = target_logits[:, pos, :]  # [batch_size, vocab_size]
                    draft_pos_logits = draft_logits[:, pos, :]    # [batch_size, vocab_size]
                    
                    # Sample from target model with temperature
                    target_probs = F.softmax(target_pos_logits / temp, dim=-1)
                    sampled_tokens = torch.multinomial(target_probs, num_samples=1).squeeze(-1)  # [batch_size]
                    
                    # Get argmax from draft model
                    draft_argmax = torch.argmax(draft_pos_logits, dim=-1)  # [batch_size]
                    
                    # Find positions where they disagree
                    disagreements = (sampled_tokens != draft_argmax)
                    disagreement_indices = torch.where(disagreements)[0]
                    
                    if len(disagreement_indices) > 0:
                        total_disagreements += len(disagreement_indices)
                        
                        # For disagreeing positions, check if sampled token is in top-F of draft
                        for idx in disagreement_indices:
                            draft_logits_for_seq = draft_pos_logits[idx]  # [vocab_size]
                            sampled_token = sampled_tokens[idx].item()
                            
                            # Get top-F tokens from draft
                            top_f_values, top_f_indices = torch.topk(draft_logits_for_seq, f)
                            
                            # Check if sampled token is in top-F
                            if sampled_token in top_f_indices:
                                overlap_count += 1
                
                # Calculate overlap fraction
                overlap_fraction = overlap_count / total_disagreements if total_disagreements > 0 else 0.0
                results[(temp, f)] = {
                    'overlap_fraction': overlap_fraction,
                    'total_disagreements': total_disagreements,
                    'overlap_count': overlap_count
                }
                
                pbar.update(1)
    
    # Print results in a neat table
    print("\n" + "="*80)
    print("TEMPERATURE vs TOP-F OVERLAP ANALYSIS")
    print("="*80)
    print("Shows fraction of disagreement positions where target sample is in draft's top-F")
    print()
    
    # Print header
    print(f"{'Temp':<6}", end="")
    for f in f_values:
        print(f"{'F=' + str(f):<8}", end="")
    print()
    print("-" * (6 + 8 * len(f_values)))
    
    # Print results
    for temp in temperatures:
        print(f"{temp:<6.1f}", end="")
        for f in f_values:
            overlap_frac = results[(temp, f)]['overlap_fraction']
            print(f"{overlap_frac:<8.3f}", end="")
        print()
    
    print()
    print("Summary Statistics:")
    print(f"Total positions analyzed: {batch_size * (seq_len - 1)}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    
    # Print some detailed stats for a few cases
    print("\nDetailed breakdown for selected (temp, F) pairs:")
    selected_cases = [(0.5, 2), (1.0, 4), (1.5, 8)]
    for temp, f in selected_cases:
        if (temp, f) in results:
            stats = results[(temp, f)]
            print(f"Temp={temp}, F={f}: {stats['overlap_count']}/{stats['total_disagreements']} = {stats['overlap_fraction']:.3f}")

if __name__ == "__main__":
    main()
