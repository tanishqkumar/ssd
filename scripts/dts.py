import os
import sys
import argparse
import json
import torch
import pickle
from random import seed

sys.path.append(os.path.dirname(__file__))
from plot_helpers import (
    get_model_paths_and_tp,
    load_models,
    prepare_batch_data,
    get_model_logits,
    analyze_top_f_overlap,
    aggregate_results,
    print_results,
    plot_results,
)


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


def get_pythia_model_paths():
    """Get Pythia model paths based on sizes."""
    base_path = "/data/tkumar/huggingface/hub"
    
    # Always use 12B as target and 0.41B as draft
    target_model_dir = "models--EleutherAI--pythia-12b"
    draft_model_dir = "models--EleutherAI--pythia-410m"
    
    # Find the snapshot directories (there should be one per model)
    target_base = os.path.join(base_path, target_model_dir, "snapshots")
    draft_base = os.path.join(base_path, draft_model_dir, "snapshots")
    
    if not os.path.exists(target_base):
        raise FileNotFoundError(f"Target model directory not found: {target_base}")
    if not os.path.exists(draft_base):
        raise FileNotFoundError(f"Draft model directory not found: {draft_base}")
    
    # Get the snapshot hash (should be only one directory)
    target_snapshots = os.listdir(target_base)
    draft_snapshots = os.listdir(draft_base)
    
    if len(target_snapshots) != 1:
        raise ValueError(f"Expected exactly one snapshot for target model, found: {target_snapshots}")
    if len(draft_snapshots) != 1:
        raise ValueError(f"Expected exactly one snapshot for draft model, found: {draft_snapshots}")
    
    target_path = os.path.join(target_base, target_snapshots[0])
    draft_path = os.path.join(draft_base, draft_snapshots[0])
    
    return target_path, draft_path


def run_analysis_sweep(target_model, draft_model, tokenizer, prompts, temps, draft_sizes, 
                      target_size, num_seqs, seq_len, device, disagreements_only, use_topf, f_values):
    """Run analysis sweep for given configuration."""
    all_results = {}
    batch_size = 16  # Fixed batch size for GPU memory management
    
    for dsize in draft_sizes:
        for temp in temps:
            print(f"\n{'-'*60}")
            print(f"RUNNING ANALYSIS: Temperature={temp}, Target={target_size}B, Draft={dsize}B")
            print(f"Disagreements only: {disagreements_only}, Use top-F: {use_topf}")
            print(f"{'-'*60}")

            # Calculate number of batches needed
            num_batches = (num_seqs + batch_size - 1) // batch_size
            all_batch_results = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_seqs)
                current_batch_size = end_idx - start_idx
                
                print(f"Processing batch {batch_idx + 1}/{num_batches} (sequences {start_idx}-{end_idx-1})")
                
                # Prepare batch data
                batch_prompts = prompts[start_idx:end_idx]
                input_ids, mask = prepare_batch_data(batch_prompts, current_batch_size, seq_len, device, tokenizer)
                
                # Get model logits for this batch
                target_logits, draft_logits = get_model_logits(target_model, draft_model, input_ids, mask)
                
                # Analyze this batch
                batch_results = analyze_top_f_overlap(target_logits, draft_logits, temp, f_values, 
                                                    seq_len, mask, disagreements_only, use_topf)
                all_batch_results.append(batch_results)
                
                # Clean up batch tensors
                del input_ids, mask, target_logits, draft_logits
                torch.cuda.empty_cache()
            
            # Aggregate results across all batches
            aggregated_results = aggregate_results(all_batch_results, f_values)
            all_results[(temp, dsize)] = aggregated_results
            
            # Print results for this configuration
            print_results(aggregated_results, temp, target_size, dsize, num_seqs, seq_len, f_values, disagreements_only, use_topf)
    
    return all_results


def main():
    # Set random seed for reproducibility
    seed(42)
    torch.manual_seed(42)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze temperature vs top-F overlap with dual sweeps')
    parser.add_argument('--temp', type=float, nargs='+', default=[0.7], help='Temperature(s) for sampling (default: [0.7], use 0.0 for greedy)')
    parser.add_argument('--size', type=int, default=8, help='Target model size in billions (default: 8)')
    parser.add_argument('--dsize', type=int, nargs='+', default=[1], help='Draft model size(s) in billions (default: [1])')
    parser.add_argument('--c4', action='store_true', help='Use C4 prompts instead of GSM8K (default: GSM8K)')
    parser.add_argument('--numseqs', type=int, default=128, help='Total number of sequences to analyze (default: 64)')
    parser.add_argument('--seqlen', type=int, default=256, help='Sequence length and prompt max length (default: 256)')
    parser.add_argument('--loglog', action='store_true', help='Use log-log scale for plotting (default: log-x, linear-y)')
    parser.add_argument('--invert', action='store_true', help='Plot rejection rate (1 - cache hit rate) instead of cache hit rate')
    parser.add_argument('--pythia', action='store_true', help='Use Pythia models instead of Llama models')
    args = parser.parse_args()
    
    temps = args.temp
    target_size = args.size
    draft_sizes = args.dsize
    num_seqs = args.numseqs
    seq_len = args.seqlen
    loglog = args.loglog
    invert = args.invert
    use_pythia = args.pythia
    
    # Set device to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define F values for analysis
    f_values = [2, 4, 8, 16, 32, 64, 128]  # for a bunch of temps, compute max(p-q,0)
    # for topf in draft sum residual probass and plot wrt F -- temp match for p-q
    
    # Store results for both sweeps
    all_results_sweep1 = {}  # topf=True, disagreements_only=False
    all_results_sweep2 = {}  # topf=False, disagreements_only=True
    
    # Load target model once at the beginning
    print(f"\n{'='*80}")
    print(f"LOADING TARGET MODEL: {target_size}B")
    print(f"{'='*80}")
    
    # Clear cache before loading models
    torch.cuda.empty_cache()
    
    # Get target model path (use first draft size just to get the target path)
    if use_pythia:
        target_path, _ = get_pythia_model_paths()
    else:
        target_path, _ = get_model_paths_and_tp(target_size, draft_sizes[0])
    
    # Load target model and tokenizer once
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading target model from: {target_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_path, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    target_model.eval()
    
    # Load dataset once using the tokenizer
    prompts = get_prompts(tokenizer, target_path, num_seqs, seq_len, c4=args.c4)
    print(f'Loaded {len(prompts)} prompts')
    
    # Ensure we have enough prompts
    if len(prompts) < num_seqs:
        print(f"Warning: Only {len(prompts)} prompts available, but {num_seqs} requested")
        num_seqs = len(prompts)
    
    # Run analysis for each draft size (reloading only draft models)
    for dsize in draft_sizes:
        print(f"\n{'='*80}")
        print(f"LOADING DRAFT MODEL: {dsize}B")
        print(f"{'='*80}")
        
        # Get draft model path
        if use_pythia:
            _, draft_path = get_pythia_model_paths()
        else:
            _, draft_path = get_model_paths_and_tp(target_size, dsize)
        
        # Load only the draft model for this size
        print(f"Loading draft model from: {draft_path}")
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_path, dtype=torch.bfloat16, device_map="auto")
        draft_model.eval()

        # Run Sweep 1: topf=True, disagreements_only=False
        print(f"\n{'='*80}")
        print("SWEEP 1: top-F analysis on all valid positions")
        print(f"{'='*80}")
        sweep1_results = run_analysis_sweep(target_model, draft_model, tokenizer, prompts, 
                                          temps, [dsize], target_size, num_seqs, seq_len, 
                                          device, disagreements_only=False, use_topf=True, f_values=f_values)
        
        # Store sweep 1 results
        for key, value in sweep1_results.items():
            all_results_sweep1[key] = value
        
        # Clear sweep 1 results from memory to save space
        del sweep1_results
        torch.cuda.empty_cache()
        
        # Run Sweep 2: topf=False, disagreements_only=True  
        print(f"\n{'='*80}")
        print("SWEEP 2: top-(F+1)\\top-1 analysis on disagreement positions")
        print(f"{'='*80}")
        sweep2_results = run_analysis_sweep(target_model, draft_model, tokenizer, prompts,
                                          temps, [dsize], target_size, num_seqs, seq_len,
                                          device, disagreements_only=True, use_topf=False, f_values=f_values)
        
        # Store sweep 2 results
        for key, value in sweep2_results.items():
            all_results_sweep2[key] = value
        
        # Clear sweep 2 results from memory to save space
        del sweep2_results
        torch.cuda.empty_cache()
        
        # Clean up draft model after finishing both sweeps for this draft size
        del draft_model
        torch.cuda.empty_cache()
    
    # Clean up target model, tokenizer, and prompts after all experiments
    del target_model, tokenizer, prompts
    torch.cuda.empty_cache()
    
    # Prepare all data for saving
    plot_data = {
        'all_results_sweep1': all_results_sweep1,
        'all_results_sweep2': all_results_sweep2,
        'target_size': target_size,
        'f_values': f_values,
        'temps': temps,
        'draft_sizes': draft_sizes,
        'num_seqs': num_seqs,
        'seq_len': seq_len,
        'loglog': loglog,
        'invert': invert,
        'use_pythia': use_pythia,
        'use_c4': args.c4
    }
    
    # Save all data to pickle file
    pickle_path = 'dts_plot_data.pkl'
    print(f"\nSaving all plot data to {pickle_path}")
    with open(pickle_path, 'wb') as f:
        pickle.dump(plot_data, f)
    print(f"Plot data saved successfully!")
    
    # Create combined side-by-side plots
    print(f"\n{'='*80}")
    print("CREATING COMPARISON PLOTS")
    print(f"{'='*80}")
    plot_results(all_results_sweep1, all_results_sweep2, target_size, f_values, loglog=loglog, invert=invert)
    
    # Final cleanup
    del all_results_sweep1, all_results_sweep2, plot_data
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
