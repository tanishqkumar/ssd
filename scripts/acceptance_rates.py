import os
import sys
import argparse
from typing import Dict, Tuple

import torch

sys.path.append(os.path.dirname(__file__))
from plot_helpers import (
    get_model_paths_and_tp,
    load_models,
    prepare_batch_data,
    get_model_logits,
    load_prompts_from_jsonl,
    gather_last_valid_logits,
    plot_acceptance_rates,
    compute_l1_acceptance_rate_first,
    compute_l1_acceptance_rate_all_positions,
)


def main():
    parser = argparse.ArgumentParser(description='Compute acceptance rates for speculative decoding')
    parser.add_argument('--temp', type=float, nargs='+', default=[0.7])
    parser.add_argument('--size', type=int, default=8, help='Target model size in billions')
    parser.add_argument('--dsize', type=int, nargs='+', default=[1, 3], help='Draft model size(s) in billions')
    parser.add_argument('--c4', action='store_true', help='Use C4 prompts instead of GSM8K')
    parser.add_argument('--numseqs', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=256)
    args = parser.parse_args()

    temps = args.temp
    target_size = args.size
    draft_sizes = args.dsize
    num_seqs = args.numseqs
    seq_len = args.seqlen
    use_c4 = args.c4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results_l1: Dict[Tuple[float, int], float] = {}
    for dsize in draft_sizes:
        print(f"\nLoading models: Target={target_size}B, Draft={dsize}B")

        target_path, draft_path = get_model_paths_and_tp(target_size, dsize)
        target_model, draft_model, tokenizer = load_models(target_path, draft_path, device)

        prompts = load_prompts_from_jsonl(tokenizer, num_seqs, seq_len, use_c4=use_c4)
        input_ids, mask = prepare_batch_data(prompts, len(prompts), seq_len, device, tokenizer)

        for temp in temps:
            with torch.inference_mode():
                lp_all, lq_all = get_model_logits(target_model, draft_model, input_ids, mask)

            lp_first = gather_last_valid_logits(lp_all, mask)
            lq_first = gather_last_valid_logits(lq_all, mask)
            B = lp_first.shape[0]
            temps_t = torch.full((B,), float(temp), device=device)
            temps_q = torch.zeros((B,), device=device)
            a_first = compute_l1_acceptance_rate_first(lp_first, lq_first, temps_t, temps_q)
            a_all = compute_l1_acceptance_rate_all_positions(lp_all, lq_all, mask, temps_t, temps_q)

            print(f"  T={temp}: first-step={a_first:.4f}, all-positions={a_all:.4f}")
            results_l1[(temp, dsize)] = a_first

        del target_model, draft_model
        torch.cuda.empty_cache()

    plot_acceptance_rates(results_l1, temps, draft_sizes, target_size, use_c4)


if __name__ == '__main__':
    main()
