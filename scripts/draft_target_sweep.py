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
    prepare_batch_data,
    get_model_logits,
    analyze_top_f_overlap,
    aggregate_results,
    print_results,
    plot_results,
)


def get_prompts(tokenizer, num_prompts, seq_len, c4=False):
    if c4:
        file_path = "/data/tkumar/huggingface//processed_datasets/c4/c4_data_5000.jsonl"
        if not os.path.exists(file_path):
            print(f"C4 file not found at {file_path}, defaulting to GSM8K")
            c4 = False

    if not c4:
        file_path = "/data/tkumar/huggingface//processed_datasets/gsm8k/gsm8k_data_10000.jsonl"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GSM8K file not found at {file_path}")

    prompts = []
    with open(file_path, 'r') as f:
        for line in f:
            if len(prompts) >= num_prompts:
                break
            data = json.loads(line.strip())
            tokens = tokenizer.encode(data["text"], add_special_tokens=False)
            target_len = max(len(tokens), 256)
            prompts.append(tokens[:target_len])

    return prompts


def run_analysis_sweep(target_model, draft_model, tokenizer, prompts, temps, draft_sizes,
                       target_size, num_seqs, seq_len, device, disagreements_only, use_topf, f_values):
    all_results = {}
    batch_size = 16

    for dsize in draft_sizes:
        for temp in temps:
            print(f"\nAnalysis: T={temp}, Target={target_size}B, Draft={dsize}B, "
                  f"disagreements_only={disagreements_only}, use_topf={use_topf}")

            num_batches = (num_seqs + batch_size - 1) // batch_size
            all_batch_results = []

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_seqs)
                current_batch_size = end_idx - start_idx

                batch_prompts = prompts[start_idx:end_idx]
                input_ids, mask = prepare_batch_data(batch_prompts, current_batch_size, seq_len, device, tokenizer)
                target_logits, draft_logits = get_model_logits(target_model, draft_model, input_ids, mask)

                batch_results = analyze_top_f_overlap(target_logits, draft_logits, temp, f_values,
                                                      seq_len, mask, disagreements_only, use_topf)
                all_batch_results.append(batch_results)

                del input_ids, mask, target_logits, draft_logits
                torch.cuda.empty_cache()

            aggregated_results = aggregate_results(all_batch_results, f_values)
            all_results[(temp, dsize)] = aggregated_results
            print_results(aggregated_results, temp, target_size, dsize, num_seqs, seq_len, f_values, disagreements_only, use_topf)

    return all_results


def main():
    seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description='Analyze temperature vs top-F overlap with dual sweeps')
    parser.add_argument('--temp', type=float, nargs='+', default=[0.7])
    parser.add_argument('--size', type=int, default=8, help='Target model size in billions')
    parser.add_argument('--dsize', type=int, nargs='+', default=[1], help='Draft model size(s) in billions')
    parser.add_argument('--c4', action='store_true', help='Use C4 prompts instead of GSM8K')
    parser.add_argument('--numseqs', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=256)
    parser.add_argument('--loglog', action='store_true', help='Use log-log scale for plotting')
    parser.add_argument('--invert', action='store_true', help='Plot rejection rate instead of cache hit rate')
    args = parser.parse_args()

    temps = args.temp
    target_size = args.size
    draft_sizes = args.dsize
    num_seqs = args.numseqs
    seq_len = args.seqlen

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    f_values = [2, 4, 8, 16, 32, 64, 128]

    all_results_sweep1 = {}
    all_results_sweep2 = {}

    torch.cuda.empty_cache()

    target_path, _ = get_model_paths_and_tp(target_size, draft_sizes[0])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading target model from: {target_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_path, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    target_model.eval()

    prompts = get_prompts(tokenizer, num_seqs, seq_len, c4=args.c4)
    print(f'Loaded {len(prompts)} prompts')

    if len(prompts) < num_seqs:
        print(f"Warning: Only {len(prompts)} prompts available, but {num_seqs} requested")
        num_seqs = len(prompts)

    for dsize in draft_sizes:
        _, draft_path = get_model_paths_and_tp(target_size, dsize)
        print(f"Loading draft model from: {draft_path}")
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_path, dtype=torch.bfloat16, device_map="auto")
        draft_model.eval()

        # sweep 1: top-F analysis on all valid positions
        sweep1_results = run_analysis_sweep(target_model, draft_model, tokenizer, prompts,
                                            temps, [dsize], target_size, num_seqs, seq_len,
                                            device, disagreements_only=False, use_topf=True, f_values=f_values)
        for key, value in sweep1_results.items():
            all_results_sweep1[key] = value
        del sweep1_results
        torch.cuda.empty_cache()

        # sweep 2: top-(F+1)\top-1 analysis on disagreement positions
        sweep2_results = run_analysis_sweep(target_model, draft_model, tokenizer, prompts,
                                            temps, [dsize], target_size, num_seqs, seq_len,
                                            device, disagreements_only=True, use_topf=False, f_values=f_values)
        for key, value in sweep2_results.items():
            all_results_sweep2[key] = value
        del sweep2_results
        torch.cuda.empty_cache()

        del draft_model
        torch.cuda.empty_cache()

    del target_model, tokenizer, prompts
    torch.cuda.empty_cache()

    plot_data = {
        'all_results_sweep1': all_results_sweep1,
        'all_results_sweep2': all_results_sweep2,
        'target_size': target_size,
        'f_values': f_values,
        'temps': temps,
        'draft_sizes': draft_sizes,
        'num_seqs': num_seqs,
        'seq_len': seq_len,
        'loglog': args.loglog,
        'invert': args.invert,
        'use_c4': args.c4
    }

    pickle_path = 'dts_plot_data.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(plot_data, f)
    print(f"Saved plot data to {pickle_path}")

    plot_results(all_results_sweep1, all_results_sweep2, target_size, f_values,
                 loglog=args.loglog, invert=args.invert)


if __name__ == "__main__":
    main()
