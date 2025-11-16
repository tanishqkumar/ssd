import os
import pickle
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def load_c4_prompts(cache_dir, model_path, num_prompts=1000, max_length=1024):
    """Load and tokenize C4 dataset snippets, with caching."""
    if load_dataset is None or AutoTokenizer is None:
        raise ImportError(
            "Required libraries (datasets, transformers) are not available. Please install them.")

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
            "allenai/c4", "en", split=f"train[:{num_prompts}]", streaming=False)
    except:
        raise ImportError("datasets library not available. Please install with: pip install datasets")

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


def prepare_batch_data(c4_prompts, batch_size, seq_len, device, tokenizer):
    """Prepare batch of input sequences from C4 prompts with padding."""
    batch_prompts = []
    assert len(c4_prompts) >= batch_size, "Not enough prompts to fill batch"

    # Get padding token ID
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for i in range(batch_size):
        prompt = c4_prompts[i]
        if len(prompt) >= seq_len:
            # Truncate to seq_len
            batch_prompts.append(prompt[:seq_len])
        else:
            # Pad to seq_len
            padded_prompt = prompt + [pad_token_id] * (seq_len - len(prompt))
            batch_prompts.append(padded_prompt)

    # Convert to tensor and move to device, ensuring the tensor is of type long for embedding layers
    input_ids = torch.tensor(batch_prompts, dtype=torch.long, device=device)

    # Create attention mask to ignore padding tokens
    attention_mask = (input_ids != pad_token_id).long()

    print(f"Input batch shape: {input_ids.shape}")
    print(f"Input tensor device: {input_ids.device}")
    print(f"Input tensor dtype: {input_ids.dtype}")
    print(f"Padding token ID: {pad_token_id}")

    return input_ids, attention_mask


def get_model_logits(target_model, draft_model, input_ids, attention_mask):
    """Get logits from both models."""
    print("Running forward passes...")
    assert input_ids.numel() > 0, "Input ids is empty"
    print(f"Input ids shape: {input_ids.shape}")
    print(f"Input ids device: {input_ids.device}")
    print(f"Input ids dtype: {input_ids.dtype}")
    with torch.inference_mode():
        target_outputs = target_model(input_ids, attention_mask=attention_mask, use_cache=False)
        draft_outputs = draft_model(input_ids, attention_mask=attention_mask, use_cache=False)

    target_logits = target_outputs.logits  # [batch_size, seq_len, vocab_size]
    draft_logits = draft_outputs.logits    # [batch_size, seq_len, vocab_size]

    print(f"Target logits shape: {target_logits.shape}")
    print(f"Draft logits shape: {draft_logits.shape}")
    print(f"Target logits device: {target_logits.device}")
    print(f"Draft logits device: {draft_logits.device}")

    return target_logits, draft_logits


def analyze_top_f_overlap(target_logits, draft_logits, temp, f_values, seq_len, attention_mask, disagreements_only=True, use_topf=True):
    """Analyze overlap between target samples and draft top-F tokens, ignoring padded positions."""
    print(f'Using {attention_mask.numel()} tokens in our analysis (non-empty portion of mask)')
    batch_size = target_logits.shape[0]
    results = {}

    print(f"Analyzing temperature {temp} with F values: {f_values}")
    print(f"Disagreements only: {disagreements_only}, Use top-F: {use_topf}")

    # Validate expected sequence length matches logits shape
    assert target_logits.shape[1] == seq_len and draft_logits.shape[1] == seq_len, \
        "Mismatch between provided seq_len and logits tensor shapes"

    # Consider positions predicting next token (exclude last position)
    target_slice = target_logits[:, :seq_len - 1, :]  # [B, S-1, V]
    draft_slice = draft_logits[:, :seq_len - 1, :]    # [B, S-1, V]

    # Create mask for valid (non-padded) positions
    # We need to check both current position and next position are valid
    valid_positions = attention_mask[:, :seq_len - 1] & attention_mask[:, 1:seq_len]  # [B, S-1]

    # Sample tokens from target model
    if temp == 0.0:
        sampled_tokens = torch.argmax(target_slice, dim=-1)  # [B, S-1]
    else:
        target_probs = F.softmax(target_slice / temp, dim=-1)  # [B, S-1, V]
        flat_probs = target_probs.reshape(-1, target_probs.size(-1))  # [(B*(S-1)), V]
        sampled_tokens = torch.multinomial(flat_probs, num_samples=1).view(
            target_probs.size(0), target_probs.size(1)
        )  # [B, S-1]

    # Draft argmax tokens
    draft_argmax = torch.argmax(draft_slice, dim=-1)  # [B, S-1]

    if disagreements_only:
        # Only analyze positions where target and draft disagree
        analysis_mask = (sampled_tokens != draft_argmax) & valid_positions  # [B, S-1]
        total_positions = int(analysis_mask.sum().item())
        analysis_desc = "disagreement positions"

        if total_positions == 0:
            # Return zero results instead of raising error
            results = {}
            for f in f_values:
                results[f] = {
                    'overlap_fraction': 0.0,
                    'total_positions': 0,
                    'overlap_count': 0
                }
            return results
    else:
        # Analyze all valid positions
        analysis_mask = valid_positions  # [B, S-1]
        total_positions = int(analysis_mask.sum().item())
        analysis_desc = "valid positions"

    # Compute top-k indices for the appropriate range
    max_f = max(f_values)
    if use_topf:
        # Use top-F directly
        topk_size = max_f
        topk_indices = torch.topk(draft_slice, k=topk_size, dim=-1).indices  # [B, S-1, topk_size]
        eq_topk = (topk_indices == sampled_tokens.unsqueeze(-1))  # [B, S-1, topk_size]
    else:
        # Use top-(F+1) excluding top-1
        topk_size = max_f + 1
        topk_indices = torch.topk(draft_slice, k=topk_size, dim=-1).indices  # [B, S-1, topk_size]
        # Exclude the top-1 (first element) when checking overlap
        eq_topk = (topk_indices[:, :, 1:] == sampled_tokens.unsqueeze(-1))  # [B, S-1, topk_size-1]

    with tqdm(total=len(f_values), desc=f"Analyzing F values at temp={temp}") as pbar:
        for f in f_values:
            if use_topf:
                # Check if target sample is in draft's top-F
                in_top_f = eq_topk[:, :, :f].any(dim=-1)  # [B, S-1]
            else:
                # Check if target sample is in draft's top-(F+1) excluding top-1
                # This gives us F positions shifted by one
                in_top_f = eq_topk[:, :, :f].any(dim=-1)  # [B, S-1]

            # Only count overlaps at analysis positions
            overlap_count = int((in_top_f & analysis_mask).sum().item())
            overlap_fraction = overlap_count / total_positions if total_positions > 0 else 0.0

            results[f] = {
                'overlap_fraction': overlap_fraction,
                'total_positions': total_positions,
                'overlap_count': overlap_count
            }

            pbar.update(1)

    return results


def aggregate_results(all_batch_results, f_values):
    """Aggregate results across multiple batches."""
    aggregated = {}

    for f in f_values:
        total_overlap_count = 0
        total_positions = 0

        for batch_results in all_batch_results:
            if f in batch_results:
                total_overlap_count += batch_results[f]['overlap_count']
                total_positions += batch_results[f]['total_positions']

        overlap_fraction = total_overlap_count / total_positions if total_positions > 0 else 0.0

        aggregated[f] = {
            'overlap_fraction': overlap_fraction,
            'total_positions': total_positions,
            'overlap_count': total_overlap_count
        }

    return aggregated


def print_results(results, temp, target_size, dsize, num_seqs, seq_len, f_values, disagreements_only=True, use_topf=True):
    """Print analysis results in a formatted table."""
    print("\n" + "="*70)
    analysis_type = "disagreement" if disagreements_only else "all valid"
    topf_type = "top-F" if use_topf else "top-(F+1) \\ top-1"
    print(f"TOP-F OVERLAP ANALYSIS (Temperature = {temp}, Target = {target_size}B, Draft = {dsize}B)")
    print(f"Analysis: {analysis_type} positions, Using: {topf_type}")
    print("="*70)
    if disagreements_only:
        if temp == 0.0:
            print("Shows fraction of disagreement positions where target greedy choice is in draft's selected set")
        else:
            print("Shows fraction of disagreement positions where target sample is in draft's selected set")
    else:
        if temp == 0.0:
            print("Shows fraction of all valid positions where target greedy choice is in draft's selected set")
        else:
            print("Shows fraction of all valid positions where target sample is in draft's selected set")
    print()

    print(f"{'F':<6}{'Overlap Fraction':<20}{'Total Positions':<15}{'Overlaps':<10}")
    print("-" * 60)

    for f in f_values:
        stats = results[f]
        print(f"{f:<6}{stats['overlap_fraction']:<20.3f}{stats['total_positions']:<15}{stats['overlap_count']:<10}")

    print()
    print("Summary Statistics:")
    print(f"Temperature: {temp} {'(greedy)' if temp == 0.0 else ''}")
    print(f"Target model size: {target_size}B, Draft model size: {dsize}B")
    print(f"Analysis type: {analysis_type} positions")
    print(f"Draft selection: {topf_type}")
    print(f"Total sequences analyzed: {num_seqs}")
    print(f"Sequence length: {seq_len}")


def load_prompts_from_jsonl(tokenizer, num_prompts, seq_len, use_c4=False):
    """Load prompts from local JSONL files (C4 or GSM8K), tokenized without padding.

    Returns a list[list[int]] of token ids. Padding/truncation is handled later.
    """
    if use_c4:
        print("Loading C4 dataset...")
        file_path = "/data/tkumar/huggingface/c4/c4_data_5000.jsonl"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"C4 file not found at {file_path}")
    else:
        print("Loading GSM8K dataset...")
        file_path = "/data/tkumar/huggingface/gsm8k/gsm8k_data_all.jsonl"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"GSM8K file not found at {file_path}")

    prompts = []
    with open(file_path, 'r') as f:
        for line in f:
            if len(prompts) >= num_prompts:
                break
            data = json.loads(line.strip())
            text = data["text"]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            prompts.append(tokens)

    return prompts


def gather_last_valid_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Gather logits at the last valid (non-pad) position per sequence.

    Args:
        logits: [B, S, V]
        attention_mask: [B, S] with 1 for valid tokens, 0 for padding

    Returns:
        last_logits: [B, V]
    """
    lengths = attention_mask.sum(dim=1).clamp(min=1)
    last_indices = (lengths - 1).to(dtype=torch.long)
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    last_logits = logits[batch_idx, last_indices, :]
    return last_logits


def sample_speculated_tokens_from_logits(logits_slice: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample speculated tokens from draft logits for K=1 speculative step.

    Args:
        logits_slice: [B, V] tensor of logits at next-token position
        temperature: scalar temperature for sampling

    Returns:
        tokens: [B] int64 tensor of sampled token ids
    """
    if temperature == 0.0:
        return logits_slice.argmax(dim=-1)
    t = max(1e-8, float(temperature))
    probs = torch.softmax((logits_slice / t).to(torch.float32), dim=-1)
    tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    return tokens


def k_step_speculate_and_logits(
    target_model,
    draft_model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    K: int,
    temp_draft: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Do K-step speculative sampling with the draft and collect matching target/draft logits.

    Args:
        target_model: HF causal LM
        draft_model: HF causal LM
        tokenizer: tokenizer with pad/eos ids
        input_ids: [B, S] long
        attention_mask: [B, S] long (1=valid)
        K: lookahead steps
        temp_draft: temperature for draft sampling

    Returns:
        logits_p: [B, K, V] target logits at each speculative step
        logits_q: [B, K, V] draft logits at each speculative step
        speculated_tokens: [B, K] sampled draft tokens
    """
    device = input_ids.device
    B, S = input_ids.shape

    # Determine pad token
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Pre-allocate extended tensors up to S+K
    S_ext = S + K
    input_ext = torch.full((B, S_ext), pad_token_id, dtype=torch.long, device=device)
    mask_ext = torch.zeros((B, S_ext), dtype=torch.long, device=device)
    input_ext[:, :S] = input_ids
    mask_ext[:, :S] = attention_mask

    lengths = attention_mask.sum(dim=1).to(dtype=torch.long)  # [B]

    # Vocab size from a quick forward pass or infer from model config
    with torch.inference_mode():
        sample_out = draft_model(input_ids[:, :1], attention_mask=attention_mask[:, :1], use_cache=False)
    V = sample_out.logits.shape[-1]

    logits_p = torch.zeros((B, K, V), dtype=sample_out.logits.dtype, device=device)
    logits_q = torch.zeros((B, K, V), dtype=sample_out.logits.dtype, device=device)
    specs = torch.zeros((B, K), dtype=torch.long, device=device)

    # Iterate K speculative steps
    for j in range(K):
        # Current input is up to lengths + j tokens per row; using mask_ext to compute last index
        with torch.inference_mode():
            out_q = draft_model(input_ext, attention_mask=mask_ext, use_cache=False)
            out_p = target_model(input_ext, attention_mask=mask_ext, use_cache=False)

        last_logits_q = gather_last_valid_logits(out_q.logits, mask_ext)  # [B, V]
        last_logits_p = gather_last_valid_logits(out_p.logits, mask_ext)  # [B, V]

        logits_p[:, j, :] = last_logits_p
        logits_q[:, j, :] = last_logits_q

        # Sample a draft token and append
        tokens_j = sample_speculated_tokens_from_logits(last_logits_q, temp_draft)  # [B]
        specs[:, j] = tokens_j

        # Write token at next position (index lengths + j)
        next_pos = lengths + j
        input_ext[torch.arange(B, device=device), next_pos] = tokens_j
        mask_ext[torch.arange(B, device=device), next_pos] = 1

    return logits_p, logits_q, specs


def compute_l1_acceptance_rate(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    temps_t: torch.Tensor,
    temps_q: torch.Tensor,
) -> float:
    """Compute acceptance rate a = (1 - ||p - q||_1)/2 averaged over all [B, K]."""
    device = logits_p.device
    B, K, V = logits_p.shape

    # Prepare p probabilities
    probs_p = torch.zeros((B, K, V), dtype=torch.float32, device=device)
    nz_p = (temps_t > 0)
    if nz_p.any():
        t = temps_t[nz_p].unsqueeze(1).unsqueeze(2).clamp(min=1e-8)
        logits_sel = logits_p[nz_p].to(torch.float32)
        probs_p[nz_p] = torch.softmax(logits_sel / t, dim=-1)
    z_p = (~nz_p)
    if z_p.any():
        argmax_p = logits_p[z_p].argmax(dim=-1)
        one_hot = torch.zeros_like(logits_p[z_p], dtype=torch.float32)
        one_hot.scatter_(2, argmax_p.unsqueeze(-1), 1.0)
        probs_p[z_p] = one_hot

    # Prepare q probabilities
    probs_q = torch.zeros((B, K, V), dtype=torch.float32, device=device)
    nz_q = (temps_q > 0)
    if nz_q.any():
        tq = temps_q[nz_q].unsqueeze(1).unsqueeze(2).clamp(min=1e-8)
        logits_sel_q = logits_q[nz_q].to(torch.float32)
        probs_q[nz_q] = torch.softmax(logits_sel_q / tq, dim=-1)
    z_q = (~nz_q)
    if z_q.any():
        argmax_q = logits_q[z_q].argmax(dim=-1)
        one_hot_q = torch.zeros_like(logits_q[z_q], dtype=torch.float32)
        one_hot_q.scatter_(2, argmax_q.unsqueeze(-1), 1.0)
        probs_q[z_q] = one_hot_q

    l1 = (probs_p - probs_q).abs().sum(dim=-1)  # [B, K]
    a = 1.0 - 0.5 * l1                          # [B, K]
    a_mean = a.mean().item()
    return float(a_mean)


def compute_l1_acceptance_rate_first(
    logits_p_first: torch.Tensor,
    logits_q_first: torch.Tensor,
    temps_t: torch.Tensor,
    temps_q: torch.Tensor,
) -> float:
    """Compute acceptance rate a = (1 - ||p - q||_1)/2 on the first step only.

    Args:
        logits_p_first: [B, V] target logits at the next-token position
        logits_q_first: [B, V] draft logits at the next-token position
        temps_t: [B] target temperatures
        temps_q: [B] draft temperatures
    """
    device = logits_p_first.device
    B, V = logits_p_first.shape

    probs_p = torch.zeros((B, V), dtype=torch.float32, device=device)
    nz_p = (temps_t > 0)
    if nz_p.any():
        t = temps_t[nz_p].unsqueeze(1).clamp(min=1e-8)
        logits_sel = logits_p_first[nz_p].to(torch.float32)
        probs_p[nz_p] = torch.softmax(logits_sel / t, dim=-1)
    z_p = (~nz_p)
    if z_p.any():
        argmax_p = logits_p_first[z_p].argmax(dim=-1)
        one_hot = torch.zeros_like(logits_p_first[z_p], dtype=torch.float32)
        one_hot.scatter_(1, argmax_p.unsqueeze(-1), 1.0)
        probs_p[z_p] = one_hot

    probs_q = torch.zeros((B, V), dtype=torch.float32, device=device)
    nz_q = (temps_q > 0)
    if nz_q.any():
        tq = temps_q[nz_q].unsqueeze(1).clamp(min=1e-8)
        logits_sel_q = logits_q_first[nz_q].to(torch.float32)
        probs_q[nz_q] = torch.softmax(logits_sel_q / tq, dim=-1)
    z_q = (~nz_q)
    if z_q.any():
        argmax_q = logits_q_first[z_q].argmax(dim=-1)
        one_hot_q = torch.zeros_like(logits_q_first[z_q], dtype=torch.float32)
        one_hot_q.scatter_(1, argmax_q.unsqueeze(-1), 1.0)
        probs_q[z_q] = one_hot_q

    l1 = (probs_p - probs_q).abs().sum(dim=-1)  # [B]
    a = 1.0 - 0.5 * l1
    return float(a.mean().item())


def compute_l1_acceptance_rate_all_positions(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    attention_mask: torch.Tensor,
    temps_t: torch.Tensor,
    temps_q: torch.Tensor,
    stripe_size: int = 8,
) -> float:
    """Compute a = (1 - ||p - q||_1)/2 averaged over all valid next-token positions,
    without allocating full [B, S, V] probability tensors. Processes sequence in stripes.

    Args:
        logits_p: [B, S, V]
        logits_q: [B, S, V]
        attention_mask: [B, S] with 1 for valid tokens, 0 for padding
        temps_t: [B]
        temps_q: [B]
        stripe_size: number of sequence positions per stripe
    """
    device = logits_p.device
    B, S, V = logits_p.shape

    valid = (attention_mask[:, :-1] & attention_mask[:, 1:]).to(dtype=torch.bool)  # [B, S-1]
    total_valid = int(valid.sum().item())
    if total_valid == 0:
        return 0.0

    nz_p = (temps_t > 0)
    z_p = (~nz_p)
    nz_q = (temps_q > 0)
    z_q = (~nz_q)

    sum_l1 = torch.zeros((), device=device, dtype=torch.float32)

    for s0 in range(0, S - 1, stripe_size):
        s1 = min(S - 1, s0 + stripe_size)
        lp = logits_p[:, s0:s1, :]
        lq = logits_q[:, s0:s1, :]
        vmask = valid[:, s0:s1]  # [B, s]

        # Case both greedy (temps==0): L1 = 0 if argmax equal else 2
        both_z = z_p & z_q
        if both_z.any():
            idx_p = lp[both_z].argmax(dim=-1)  # [Bz, s]
            idx_q = lq[both_z].argmax(dim=-1)
            eq = (idx_p == idx_q)
            l1 = torch.where(eq, torch.zeros_like(eq, dtype=torch.float32, device=device), torch.full_like(eq, 2.0, dtype=torch.float32, device=device))
            sum_l1 += (l1 * vmask[both_z].float()).sum()

        # Case p softmax, q greedy: L1 = 2*(1 - p(a_q))
        p_nz_q_z = nz_p & z_q
        if p_nz_q_z.any():
            t = temps_t[p_nz_q_z].view(-1, 1, 1).clamp(min=1e-8)
            lp_sub = (lp[p_nz_q_z].to(torch.float32) / t)  # [Bnz, s, V]
            logZp = torch.logsumexp(lp_sub, dim=-1)       # [Bnz, s]
            idx_q = lq[p_nz_q_z].argmax(dim=-1)           # [Bnz, s]
            p_logits_at_idx = lp_sub.gather(-1, idx_q.unsqueeze(-1)).squeeze(-1)  # [Bnz, s]
            p_prob = torch.exp(p_logits_at_idx - logZp)
            l1 = 2.0 * (1.0 - p_prob)
            sum_l1 += (l1 * vmask[p_nz_q_z].float()).sum()

        # Case p greedy, q softmax: L1 = 2*(1 - q(a_p))
        p_z_q_nz = z_p & nz_q
        if p_z_q_nz.any():
            tq = temps_q[p_z_q_nz].view(-1, 1, 1).clamp(min=1e-8)
            lq_sub = (lq[p_z_q_nz].to(torch.float32) / tq)
            logZq = torch.logsumexp(lq_sub, dim=-1)       # [Bnz, s]
            idx_p = lp[p_z_q_nz].argmax(dim=-1)           # [Bnz, s]
            q_logits_at_idx = lq_sub.gather(-1, idx_p.unsqueeze(-1)).squeeze(-1)
            q_prob = torch.exp(q_logits_at_idx - logZq)
            l1 = 2.0 * (1.0 - q_prob)
            sum_l1 += (l1 * vmask[p_z_q_nz].float()).sum()

        # Case both softmax: L1 = sum_i |p_i - q_i|
        both_nz = nz_p & nz_q
        if both_nz.any():
            t = temps_t[both_nz].view(-1, 1, 1).clamp(min=1e-8)
            tq = temps_q[both_nz].view(-1, 1, 1).clamp(min=1e-8)
            p_probs = torch.softmax(lp[both_nz].to(torch.float32) / t, dim=-1)  # [Bnz, s, V]
            q_probs = torch.softmax(lq[both_nz].to(torch.float32) / tq, dim=-1)
            l1 = (p_probs - q_probs).abs().sum(dim=-1)  # [Bnz, s]
            sum_l1 += (l1 * vmask[both_nz].float()).sum()

        # Free stripe tensors sooner
        del lp, lq
        torch.cuda.empty_cache()

    mean_l1 = sum_l1 / float(total_valid)
    a_mean = 1.0 - 0.5 * mean_l1
    return float(a_mean.item())


def plot_acceptance_histogram(
    hist_actual: list,
    a_hat: float,
    K: int,
    title: str,
    out_path: str,
):
    """Plot predicted vs actual histogram of acceptance lengths j in [0..K]."""
    import matplotlib.pyplot as plt
    import numpy as np

    total = sum(hist_actual)
    if total == 0:
        total = 1
    actual_freq = np.array(hist_actual, dtype=np.float64) / float(total)

    # Theoretical geometric pmf truncated at K
    j = np.arange(0, K + 1)
    pred = (a_hat ** j) * (1.0 - a_hat)
    pred[-1] = a_hat ** K
    pred = pred / pred.sum()  # Ensure sums to 1 (minor numeric fixes)

    x = np.arange(K + 1)
    width = 0.35

    plt.figure(figsize=(12, 8))  # Larger figure for better text spacing
    
    # Use blue gradient for bars per style guide
    plt.bar(x - width / 2, actual_freq, width=width, label='Actual', alpha=0.9, 
            color='#4A90E2', edgecolor='white', linewidth=1.5)  # Primary blue
    plt.bar(x + width / 2, pred, width=width, label='Predicted (geom)', alpha=0.85, 
            color='#2E5BBA', edgecolor='white', linewidth=1.5)  # Darker blue for contrast

    plt.xlabel('Accepted length j', fontsize=22, fontweight='bold')  # MASSIVE labels
    plt.ylabel('Frequency', fontsize=22, fontweight='bold')
    plt.title(title, fontsize=24, fontweight='bold', pad=25)  # HUGE title
    plt.xticks(x, [str(int(i)) for i in x], fontsize=18)  # LARGE tick labels
    plt.yticks(fontsize=18)
    plt.ylim(0, 1)
    plt.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='#F8FAFC')  # Light gray grid
    plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=16, loc='upper right')  # Large legend
    
    # Remove top and right spines per style guide
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#F8FAFC')  # Light background per style guide

    # Ensure extension is PDF
    if not out_path.endswith('.pdf'):
        out_path_pdf = out_path.rsplit('.', 1)[0] + '.pdf' if '.' in out_path else out_path + '.pdf'
    else:
        out_path_pdf = out_path
    os.makedirs(os.path.dirname(out_path_pdf), exist_ok=True)
    plt.tight_layout(pad=4.0)  # Extra padding to prevent text overlap
    plt.savefig(out_path_pdf, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    print(f"Saved histogram plot: {out_path_pdf}")


def plot_acceptance_rates(results_by_pair: dict, temps: list, draft_sizes: list, target_size: int, use_c4: bool):
    """Plot acceptance rates in two subplots: line over temps and bars over draft sizes."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not temps or not draft_sizes:
        print("No temps or draft sizes provided; skipping acceptance plot.")
        return

    temps_sorted = sorted(list(set(temps)))
    drafts_sorted = sorted(list(set(draft_sizes)))

    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))  # Much larger figure

    # Blue gradient palette per style guide
    palette = ['#4A90E2', '#6BB6FF', '#2E5BBA', '#1E3A8A', '#0F172A', 
               '#87CEEB', '#4682B4', '#1E6091', '#003366', '#000080']
    line_width = 4  # Very bold lines
    marker_size = 12  # Large markers
    alpha = 0.95

    # Build matrix of values [len(drafts) x len(temps)]
    M = np.zeros((len(drafts_sorted), len(temps_sorted)), dtype=float)
    for i, d in enumerate(drafts_sorted):
        for j, t in enumerate(temps_sorted):
            M[i, j] = float(results_by_pair.get((t, d), 0.0) or 0.0)

    # Left: acceptance vs temperature (one line per draft size)
    for i, d in enumerate(drafts_sorted):
        y = M[i, :]
        color = palette[i % len(palette)]
        ax1.plot(temps_sorted, y, '-', color=color, linewidth=line_width,
                 markersize=marker_size, label=f'{d}B draft', alpha=alpha, marker='o')
        ax1.scatter(temps_sorted, y, color=color, s=100)  # Large scatter points

    # HUGE labels per style guide
    ax1.set_xlabel('Temperature', fontsize=22, fontweight='bold')
    ax1.set_ylabel('Acceptance rate', fontsize=22, fontweight='bold')
    ax1.set_title(f'Acceptance vs Temperature\n(Target {target_size}B, {"C4" if use_c4 else "GSM8K"})',
                  fontsize=24, fontweight='bold', pad=25)  # Split title to prevent overlap
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5,
             color='#F8FAFC')  # Light gray grid
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=16, loc='best')
    ax1.set_ylim(0.0, 1.0)
    ax1.spines['top'].set_visible(False)  # Remove top spine per style guide
    ax1.spines['right'].set_visible(False)  # Remove right spine per style guide
    ax1.set_facecolor('#F8FAFC')  # Light background per style guide
    ax1.tick_params(axis='both', labelsize=18)  # LARGE tick labels

    # Right: acceptance vs draft size (grouped bars per temp)
    x = np.arange(len(drafts_sorted))
    # Adjust width to prevent overlap with many temperatures
    width = min(0.12 + 0.015 * max(0, len(temps_sorted) - 3), 0.25)
    for j, t in enumerate(temps_sorted):
        vals = M[:, j]
        offset = (j - (len(temps_sorted) - 1) / 2) * width
        color = palette[j % len(palette)]
        ax2.bar(x + offset, vals, width, label=f'T={"0" if t == 0.0 else t}',
                color=color, alpha=0.9, edgecolor='white', linewidth=1.5)

    ax2.set_xlabel('Draft size (B)', fontsize=22, fontweight='bold')
    ax2.set_ylabel('Acceptance rate', fontsize=22, fontweight='bold')
    ax2.set_title(f'Acceptance vs Draft Size\n(Target {target_size}B, {"C4" if use_c4 else "GSM8K"})',
                  fontsize=24, fontweight='bold', pad=25)  # Split title
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{d}B' for d in drafts_sorted], fontsize=18)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='-',
             linewidth=0.5, color='#F8FAFC')
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=16, loc='best')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_facecolor('#F8FAFC')
    ax2.tick_params(axis='both', labelsize=18)

    fig.patch.set_facecolor('white')  # Clean white background per style guide
    plt.tight_layout(pad=4.0)  # Extra padding to prevent text overlap

    plot_dir = '/home/tkumar/ssd/scripts/plots/alpha'
    os.makedirs(plot_dir, exist_ok=True)
    temp_strs = [f'T0' if t == 0.0 else f'T{t}' for t in temps_sorted]
    draft_strs = [f'{d}B' for d in drafts_sorted]
    plot_filename = os.path.join(
        plot_dir,
        f'alpha_acceptance_{"_".join(temp_strs)}_{"_".join(draft_strs)}_target{target_size}B_{"c4" if use_c4 else "gsm8k"}.pdf'
    )

    plt.savefig(plot_filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print(f"\nAcceptance rate plot saved as: {plot_filename}")


def plot_results(all_results_sweep1, all_results_sweep2, target_size, f_values, loglog=False, invert=False):
    """Create and save side-by-side plots of top-F overlap results for two experimental sweeps."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Set up the figure with improved styling following style guide
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))  # Wider and shorter per request

    # Professional blue color palette from style guide
    temp_colors = {0.0: '#0F172A', 0.3: '#1E3A8A',
                   0.7: '#2E5BBA', 1.0: '#4A90E2'}
    draft_styles = {1: '-', 3: '--', 8: '-.', 70: ':'}
    line_width = 3  # Bold lines per style guide
    marker_size = 10  # Large markers per style guide
    alpha = 0.95

    # Plot sweep 1 (topf=True, disagreements_only=False) - p_{hit,nn}(F)
    for (temp, dsize), results in all_results_sweep1.items():
        f_vals = list(f_values)
        overlap_fractions = [results[f]['overlap_fraction'] for f in f_vals]

        if invert:
            overlap_fractions = [1 - frac for frac in overlap_fractions]

        color = temp_colors.get(temp, '#1E293B')
        style = draft_styles.get(dsize, '-')
        temp_str = 'T=0' if temp == 0.0 else f'T={temp}'
        label = f'{temp_str}, {dsize}B draft'

        ax1.plot(f_vals, overlap_fractions, style, color=color, linewidth=line_width,
                 markersize=marker_size, label=label, alpha=alpha, marker='o')

    # Plot sweep 2 (topf=False, disagreements_only=True) - p_{hit,n}(F)
    for (temp, dsize), results in all_results_sweep2.items():
        f_vals = list(f_values)
        overlap_fractions = [results[f]['overlap_fraction'] for f in f_vals]

        if invert:
            overlap_fractions = [1 - frac for frac in overlap_fractions]

        color = temp_colors.get(temp, '#1E293B')
        style = draft_styles.get(dsize, '-')
        temp_str = 'T=0' if temp == 0.0 else f'T={temp}'
        label = f'{temp_str}, {dsize}B draft'

        ax2.plot(f_vals, overlap_fractions, style, color=color, linewidth=line_width,
                 markersize=marker_size, label=label, alpha=alpha, marker='s')

    # Plot sweep 3 (derived function) - $p_{\text{hit}}(F)$
    for (temp, dsize) in all_results_sweep1.keys():
        if (temp, dsize) in all_results_sweep2:
            f_vals = list(f_values)
            left_fractions = [
                all_results_sweep1[(temp, dsize)][f]['overlap_fraction'] for f in f_vals]
            right_fractions = [
                all_results_sweep2[(temp, dsize)][f]['overlap_fraction'] for f in f_vals]

            # Compute $p_{\text{hit}}(F)$ = left(F) / (1 + left(F) - right(F))
            x_values = []
            for left, right in zip(left_fractions, right_fractions):
                denominator = 1 + left - right
                if denominator != 0:
                    x_val = left / denominator
                else:
                    x_val = float('inf')
                x_values.append(x_val)

            color = temp_colors.get(temp, '#1E293B')
            style = draft_styles.get(dsize, '-')
            temp_str = 'T=0' if temp == 0.0 else f'T={temp}'
            label = f'{temp_str}, {dsize}B draft'

            ax3.plot(f_vals, x_values, style, color=color, linewidth=line_width,
                     markersize=marker_size, label=label, alpha=alpha, marker='^')

    # Configure first subplot - p_{hit,nn}(F)
    ylabel = 'Rejection rate' if invert else 'Hit rate'
    # ENORMOUS axis labels per style guide for paper visibility
    ax1.set_xlabel('F', fontsize=30, fontweight='bold')  # Even larger per style guide
    ax1.set_ylabel(ylabel, fontsize=30, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5,
             color='#F8FAFC')  # Light gray grid
    ax1.set_xscale('log', base=2)
    if loglog:
        ax1.set_yscale('log')
    ax1.set_xticks(f_vals)
    ax1.set_xticklabels([str(f) for f in f_vals],
                        fontsize=26)  # HUGE tick labels for paper readability
    ax1.tick_params(axis='y', labelsize=26)
    ax1.spines['top'].set_visible(False)  # Remove top spine per style guide
    # Remove right spine per style guide
    ax1.spines['right'].set_visible(False)
    ax1.set_facecolor('#F8FAFC')  # Light background per style guide

    # Configure second subplot - p_{hit,n}(F)
    ax2.set_xlabel('F', fontsize=30, fontweight='bold')
    ax2.set_ylabel(ylabel, fontsize=30, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#F8FAFC')
    ax2.set_xscale('log', base=2)
    if loglog:
        ax2.set_yscale('log')
    ax2.set_xticks(f_vals)
    ax2.set_xticklabels([str(f) for f in f_vals], fontsize=26)
    ax2.tick_params(axis='y', labelsize=26)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_facecolor('#F8FAFC')

    # Configure third subplot - $p_{\text{hit}}(F)$
    ax3.set_xlabel('F', fontsize=30, fontweight='bold')
    title3 = r'$p_{\mathrm{hit}}(F)$'
    ax3.set_ylabel(title3, fontsize=30, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#F8FAFC')
    # Only add legend to the rightmost plot per request
    ax3.legend(frameon=True, fancybox=True, shadow=True,
               fontsize=20)  # LARGE legend per style guide
    ax3.set_xscale('log', base=2)
    if loglog:
        ax3.set_yscale('log')
    ax3.set_xticks(f_vals)
    ax3.set_xticklabels([str(f) for f in f_vals], fontsize=26)
    ax3.tick_params(axis='y', labelsize=26)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_facecolor('#F8FAFC')

    # Match y-axis ranges for first two plots for better comparison
    y1_min, y1_max = ax1.get_ylim()
    y2_min, y2_max = ax2.get_ylim()
    y_min = min(y1_min, y2_min)
    y_max = max(y1_max, y2_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # Clean white background per style guide
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=4.0)  # Extra padding to prevent text overlap

    # Create plots directory if it doesn't exist
    plot_dir = '/home/tkumar/ssd/scripts/plots/camera'
    os.makedirs(plot_dir, exist_ok=True)

    # Create filename with all configurations
    temp_strs = set()
    draft_strs = set()
    for results_dict in [all_results_sweep1, all_results_sweep2]:
        for t, d in results_dict.keys():
            temp_strs.add('T0' if t == 0.0 else f'T{t}')
            draft_strs.add(f'{d}B')

    plot_filename = os.path.join(plot_dir,
                                 f'dts_analysis_{"_".join(sorted(temp_strs))}_{"_".join(sorted(draft_strs))}_target{target_size}B{"_loglog" if loglog else ""}{"_reject" if invert else ""}.pdf')

    # Save as PDF per style guide for publication quality
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    print(f"\nHigh-quality plot saved as: {plot_filename}")
