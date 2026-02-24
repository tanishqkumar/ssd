import os
import sys
import pickle
import torch
import torch.nn.functional as F
from random import seed
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def load_c4_prompts(cache_dir, model_path, num_prompts=1000, max_length=1024):
    """Load and tokenize C4 dataset snippets, with caching."""
    cache_file = os.path.join(cache_dir, f"c4_prompts_{num_prompts}_{max_length}.pkl")

    if os.path.exists(cache_file):
        print(f"Loading cached C4 prompts from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Loading C4 dataset and tokenizing {num_prompts} prompts...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    try:
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    except:
        raise ImportError("datasets library not available. Please install with: pip install datasets")

    prompts = []
    with tqdm(total=num_prompts, desc="Processing C4 dataset") as pbar:
        for i, example in enumerate(dataset):
            if i >= num_prompts:
                break
            tokens = tokenizer.encode(example['text'], max_length=max_length, truncation=True)
            if len(tokens) >= 50:
                prompts.append(tokens)
            pbar.update(1)

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(prompts, f)

    print(f"Cached {len(prompts)} tokenized prompts to {cache_file}")
    return prompts


def get_model_paths():
    cache_dir = "/data/tkumar/huggingface/hub"

    target_model_name = "Llama-3.1-70B-Instruct"
    target_path = os.path.join(cache_dir, f"models--meta-llama--{target_model_name}", "snapshots")

    draft_model_name = "Llama-3.2-1B-Instruct"
    draft_path = os.path.join(cache_dir, f"models--meta-llama--{draft_model_name}", "snapshots")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    target_path, draft_path = get_model_paths()
    cache_dir = "/data/tkumar/huggingface/hub"

    print(f"Loading target model from: {target_path}")
    print(f"Loading draft model from: {draft_path}")

    target_model = AutoModelForCausalLM.from_pretrained(target_path, torch_dtype=torch.bfloat16, tp_plan="auto")
    draft_model = AutoModelForCausalLM.from_pretrained(draft_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(target_path)

    target_model = target_model.to(device)
    draft_model = draft_model.to(device)

    c4_prompts = load_c4_prompts(cache_dir, target_path, num_prompts=100, max_length=256)

    batch_size = 32
    seq_len = 512
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    batch_prompts = []
    for i in range(batch_size):
        prompt = c4_prompts[i % len(c4_prompts)]
        if len(prompt) >= seq_len:
            prompt = prompt[:seq_len]
        else:
            prompt = prompt + [pad_token] * (seq_len - len(prompt))
        batch_prompts.append(prompt)

    input_ids = torch.tensor(batch_prompts, device=device)

    with torch.no_grad():
        target_outputs = target_model(input_ids)
        draft_outputs = draft_model(input_ids)

    target_logits = target_outputs.logits
    draft_logits = draft_outputs.logits

    temperatures = [0.1, 0.3, 0.5, 0.7, 1.0]
    f_values = [1, 2, 4, 8]

    results = {}

    total_combinations = len(temperatures) * len(f_values)
    with tqdm(total=total_combinations, desc="Analyzing temp/F combinations") as pbar:
        for temp in temperatures:
            for f in f_values:
                total_disagreements = 0
                overlap_count = 0

                for pos in range(seq_len - 1):
                    target_pos_logits = target_logits[:, pos, :]
                    draft_pos_logits = draft_logits[:, pos, :]

                    target_probs = F.softmax(target_pos_logits / temp, dim=-1)
                    sampled_tokens = torch.multinomial(target_probs, num_samples=1).squeeze(-1)

                    draft_argmax = torch.argmax(draft_pos_logits, dim=-1)

                    disagreements = (sampled_tokens != draft_argmax)
                    disagreement_indices = torch.where(disagreements)[0]

                    if len(disagreement_indices) > 0:
                        total_disagreements += len(disagreement_indices)

                        for idx in disagreement_indices:
                            draft_logits_for_seq = draft_pos_logits[idx]
                            sampled_token = sampled_tokens[idx].item()

                            top_f_values, top_f_indices = torch.topk(draft_logits_for_seq, f)

                            if sampled_token in top_f_indices:
                                overlap_count += 1

                overlap_fraction = overlap_count / total_disagreements if total_disagreements > 0 else 0.0
                results[(temp, f)] = {
                    'overlap_fraction': overlap_fraction,
                    'total_disagreements': total_disagreements,
                    'overlap_count': overlap_count
                }

                pbar.update(1)

    print("\nTEMPERATURE vs TOP-F OVERLAP ANALYSIS")
    print("Shows fraction of disagreement positions where target sample is in draft's top-F\n")

    print(f"{'Temp':<6}", end="")
    for f in f_values:
        print(f"{'F=' + str(f):<8}", end="")
    print()
    print("-" * (6 + 8 * len(f_values)))

    for temp in temperatures:
        print(f"{temp:<6.1f}", end="")
        for f in f_values:
            overlap_frac = results[(temp, f)]['overlap_fraction']
            print(f"{overlap_frac:<8.3f}", end="")
        print()

    print(f"\nTotal positions analyzed: {batch_size * (seq_len - 1)}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")

    selected_cases = [(0.5, 2), (1.0, 4), (1.5, 8)]
    for temp, f in selected_cases:
        if (temp, f) in results:
            stats = results[(temp, f)]
            print(f"Temp={temp}, F={f}: {stats['overlap_count']}/{stats['total_disagreements']} = {stats['overlap_fraction']:.3f}")


if __name__ == "__main__":
    main()
