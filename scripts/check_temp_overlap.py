import os
import sys
import argparse
import json
import torch
from random import seed
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def main():
    seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_size = 70
    draft_size = 1

    target_path, draft_path = get_model_paths_and_tp(target_size, draft_size)
    target_model, draft_model, tokenizer = load_models(target_path, draft_path, device)

    prompts = get_prompts(tokenizer, 1, 256, c4=True)
    print(f'Loaded {len(prompts)} prompts')

    if len(prompts) > 0:
        first_prompt = prompts[0]
        print(f"First prompt text: {tokenizer.decode(first_prompt)}")

        input_ids = torch.tensor([first_prompt], dtype=torch.long, device=device)

        print("Running inference...")
        with torch.no_grad():
            outputs = target_model(input_ids)
            logits = outputs.logits

        print(f"Output logits shape: {logits.shape}")


if __name__ == "__main__":
    main()
