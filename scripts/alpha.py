import os
import sys
import argparse
from typing import List, Dict, Tuple

import torch

sys.path.append(os.path.dirname(__file__))
from plot_helpers import (
    get_model_paths_and_tp,
    load_models,
    prepare_batch_data,
    get_model_logits,
    load_prompts_from_jsonl,
    gather_last_valid_logits,
    sample_speculated_tokens_from_logits,
    plot_acceptance_rates,
    k_step_speculate_and_logits,
    compute_l1_acceptance_rate,
    plot_acceptance_histogram,
    compute_l1_acceptance_rate_first,
    compute_l1_acceptance_rate_all_positions,
)

from hf_verify import verify


def compute_acceptance_for_batch(
    target_model,
    draft_model,
    tokenizer,
    prompts_batch: List[List[int]],
    device: torch.device,
    temp_target: float,
    temp_draft: float,
    seq_len: int,
) -> float:
    """Compute acceptance rate for a batch of prompts at K=1 spec step.

    Steps:
      1) Build padded input_ids/attention_mask for the prompts
      2) Run both models to get logits [B, S, V]
      3) Gather last valid position logits from both models -> [B, V]
      4) Draft: sample one token per row from [B, V] with temp_draft
      5) Build logits_p/logits_q/speculations of shape [B, K, V] / [B, K]
      6) Call verify and compute fraction accept==1
    """
    input_ids, mask = prepare_batch_data(prompts_batch, len(prompts_batch), seq_len, device, tokenizer)

    with torch.inference_mode():
        logits_p_all, logits_q_all = get_model_logits(target_model, draft_model, input_ids, mask)

    # Gather last valid logits (next-token prediction is at current last position)
    logits_p_last = gather_last_valid_logits(logits_p_all, mask)  # [B, V]
    logits_q_last = gather_last_valid_logits(logits_q_all, mask)  # [B, V]

    # Sample speculated tokens from draft
    spec_tokens = sample_speculated_tokens_from_logits(logits_q_last, temp_draft)  # [B]

    # Prepare verify inputs; K=1
    B, V = logits_p_last.shape
    logits_p = logits_p_last.unsqueeze(1)  # [B, 1, V]
    logits_q = logits_q_last.unsqueeze(1)  # [B, 1, V]
    speculations = spec_tokens.unsqueeze(1)  # [B, 1]
    temps_t = torch.full((B,), float(temp_target), device=device)
    temps_q = torch.full((B,), float(temp_draft), device=device)

    # Call verify
    accepted_suffixes, rec = verify(logits_p, logits_q, speculations, temps_t, temps_q)

    # Acceptance is 1 if len(suffix) == 1
    accept_count = sum(1 for s in accepted_suffixes if len(s) == 1)
    return accept_count / float(B)


def compute_hist_for_batch(
    target_model,
    draft_model,
    tokenizer,
    prompts_batch: List[List[int]],
    device: torch.device,
    temp_target: float,
    temp_draft: float,
    seq_len: int,
    K: int,
) -> list[int]:
    """Compute accepted length histogram counts for a batch given K-lookahead."""
    # For speculative decoding, we need to first extend sequences with K draft tokens
    input_ids, mask = prepare_batch_data(prompts_batch, len(prompts_batch), seq_len, device, tokenizer)
    
    # Step 1: Generate K draft tokens greedily
    current_ids = input_ids.clone()
    current_mask = mask.clone()
    draft_tokens = []
    
    with torch.inference_mode():
        for step in range(K):
            # Get draft model logits
            draft_outputs = draft_model(current_ids, attention_mask=current_mask)
            draft_logits = draft_outputs.logits[:, -1, :]  # [B, V]
            
            # Greedy sampling for draft (temp_draft=0.0)
            next_tokens = draft_logits.argmax(dim=-1)  # [B]
            draft_tokens.append(next_tokens)
            
            # Extend sequences
            current_ids = torch.cat([current_ids, next_tokens.unsqueeze(1)], dim=1)
            current_mask = torch.cat([current_mask, torch.ones(current_mask.size(0), 1, device=current_mask.device)], dim=1)
    
    # Stack draft tokens: [B, K]
    speculations = torch.stack(draft_tokens, dim=1)
    
    # Step 2: Get target model logits for verification
    with torch.inference_mode():
        target_outputs = target_model(current_ids, attention_mask=current_mask)
        target_logits_full = target_outputs.logits  # [B, seq_len + K, V]
        draft_outputs = draft_model(current_ids, attention_mask=current_mask)
        draft_logits_full = draft_outputs.logits  # [B, seq_len + K, V]
    
    # IMPORTANT: Get logits at positions where draft tokens were generated
    # If draft generated tokens at positions L, L+1, ..., L+K-1,
    # then target logits should be at positions L-1, L, ..., L+K-2
    original_seq_len = input_ids.size(1)
    logits_p = target_logits_full[:, original_seq_len-1:original_seq_len-1+K, :]  # [B, K, V]
    logits_q = draft_logits_full[:, original_seq_len-1:original_seq_len-1+K, :]  # [B, K, V]
    
    B = logits_p.shape[0]
    temps_t = torch.full((B,), float(temp_target), device=device)
    temps_q = torch.zeros((B,), device=device)  # Draft is greedy
    
    accepted_suffixes, _ = verify(logits_p, logits_q, speculations, temps_t, temps_q)
    hist = [0] * (K + 1)
    for s in accepted_suffixes:
        n = min(K, len(s))
        hist[n] += 1
    return hist


def run_acceptance_sweep(
    target_model,
    draft_model,
    tokenizer,
    prompts: List[List[int]],
    temps: List[float],
    draft_sizes: List[int],
    target_size: int,
    device: torch.device,
    num_seqs: int,
    seq_len: int,
    batch_size: int = 64,
    use_c4: bool = False,
) -> Dict[Tuple[float, int], float]:
    """Run acceptance-rate sweep over (temps, draft_sizes) using batched computation."""
    results: Dict[Tuple[float, int], float] = {}
    total = num_seqs

    for dsize in draft_sizes:
        for temp in temps:
            print(f"\n{'-'*60}")
            print(f"ACCEPTANCE SWEEP: T={temp}, Target={target_size}B, Draft={dsize}B")
            print(f"{'-'*60}")

            accept_sum = 0.0
            processed = 0

            # Iterate over prompts in batches
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch = prompts[start:end]
                if len(batch) == 0:
                    continue
                print(f"Processing batch {start//batch_size + 1} ({start}-{end-1})")
                acc = compute_acceptance_for_batch(
                    target_model,
                    draft_model,
                    tokenizer,
                    batch,
                    device,
                    temp_target=temp,
                    temp_draft=temp,
                    seq_len=seq_len,
                )
                accept_sum += acc * len(batch)
                processed += len(batch)
                torch.cuda.empty_cache()

            results[(temp, dsize)] = accept_sum / max(1, processed)
            print(f"Acceptance rate: {results[(temp, dsize)]:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute acceptance rates and histograms for speculative decoding')
    parser.add_argument('--temp', type=float, nargs='+', default=[0.7], help='Temperature(s) (default: [0.7], 0.0 for greedy)')
    parser.add_argument('--size', type=int, default=8, help='Target model size in billions (default: 8)')
    parser.add_argument('--dsize', type=int, nargs='+', default=[1, 3], help='Draft model size(s) in billions (default: [1,3])')
    parser.add_argument('--c4', action='store_true', help='Use C4 prompts instead of GSM8K (default: GSM8K)')
    parser.add_argument('--numseqs', type=int, default=128, help='Total number of prompts to evaluate (default: 512)')
    parser.add_argument('--seqlen', type=int, default=256, help='Max prompt length used for context (default: 256)')
    parser.add_argument('--batch', type=int, default=32, help='Batch size (default: 64)')
    parser.add_argument('--K', type=int, default=4, help='Speculative lookahead steps K (default: 4)')
    args = parser.parse_args()

    temps = args.temp
    target_size = args.size
    draft_sizes = args.dsize
    num_seqs = args.numseqs
    seq_len = args.seqlen
    batch_size = args.batch
    use_c4 = args.c4
    K = args.K

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Compute first-step L1 acceptance rate per (temp, dsize) and plot
    results_l1: Dict[Tuple[float, int], float] = {}
    for dsize in draft_sizes:
        print(f"\n{'='*80}")
        print(f"LOADING MODELS: Target={target_size}B, Draft={dsize}B")
        print(f"{'='*80}")

        target_path, draft_path = get_model_paths_and_tp(target_size, dsize)
        target_model, draft_model, tokenizer = load_models(target_path, draft_path, device)

        prompts = load_prompts_from_jsonl(tokenizer, num_seqs, seq_len, use_c4=use_c4)
        input_ids, mask = prepare_batch_data(prompts, len(prompts), seq_len, device, tokenizer)

        for temp in temps:
            with torch.inference_mode():
                lp_all, lq_all = get_model_logits(target_model, draft_model, input_ids, mask)
            # Option A: first-step acceptance (q greedy like acc_test)
            lp_first = gather_last_valid_logits(lp_all, mask)
            lq_first = gather_last_valid_logits(lq_all, mask)
            B = lp_first.shape[0]
            temps_t = torch.full((B,), float(temp), device=device)
            temps_q = torch.zeros((B,), device=device)
            a_first = compute_l1_acceptance_rate_first(lp_first, lq_first, temps_t, temps_q)

            # Option B: all valid positions average (q greedy)
            a_all = compute_l1_acceptance_rate_all_positions(lp_all, lq_all, mask, temps_t, temps_q)

            # Use first-step for the plotted rate (matches common definition and acc_test)
            results_l1[(temp, dsize)] = a_first

        del target_model, draft_model
        torch.cuda.empty_cache()

    plot_acceptance_rates(results_l1, temps, draft_sizes, target_size, use_c4)

    # Build histogram for a reference configuration: first temp and first dsize
    if len(temps) > 0 and len(draft_sizes) > 0:
        conf_temp = temps[0]
        conf_dsize = draft_sizes[0]
        target_path, draft_path = get_model_paths_and_tp(target_size, conf_dsize)
        target_model, draft_model, tokenizer = load_models(target_path, draft_path, device)
        prompts_hist = load_prompts_from_jsonl(tokenizer, min(1024, num_seqs), seq_len, use_c4=use_c4)
        # Accumulate histogram in batches to manage memory if needed
        hist_total = [0] * (K + 1)
        for start in range(0, len(prompts_hist), batch_size):
            end = min(start + batch_size, len(prompts_hist))
            batch_prompts = prompts_hist[start:end]
            hist_part = compute_hist_for_batch(
                target_model, draft_model, tokenizer, batch_prompts, device,
                temp_target=conf_temp, temp_draft=conf_temp, seq_len=seq_len, K=K
            )
            for j in range(K + 1):
                hist_total[j] += hist_part[j]

        # Also compute a_hat for this same configuration
        input_ids, mask = prepare_batch_data(prompts_hist, len(prompts_hist), seq_len, device, tokenizer)
        # For a_hat, use q greedy to match the geometric prediction baseline
        logits_p, logits_q, _ = k_step_speculate_and_logits(target_model, draft_model, tokenizer, input_ids, mask, K, 0.0)
        B = logits_p.shape[0]
        temps_t = torch.full((B,), float(conf_temp), device=device)
        temps_q = torch.zeros((B,), device=device)
        a_hat = compute_l1_acceptance_rate(logits_p, logits_q, temps_t, temps_q)

        hist_dir = '/home/tkumar/ssd/scripts/plots/alpha'
        os.makedirs(hist_dir, exist_ok=True)
        title = f'Acceptance length distribution (T={conf_temp}, Draft={conf_dsize}B, Target={target_size}B, K={K})'
        out_path = os.path.join(hist_dir, f'alpha_hist_T{conf_temp}_D{conf_dsize}B_target{target_size}B_K{K}.png')
        plot_acceptance_histogram(hist_total, a_hat, K, title, out_path)


if __name__ == '__main__':
    main()
