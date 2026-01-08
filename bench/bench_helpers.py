import os
import json
from random import randint
from typing import List, Optional, Tuple
from transformers import AutoTokenizer


# Dataset file paths on local system
DATASET_PATHS = {
    "humaneval": "/data/tkumar/huggingface/processed_datasets/humaneval/humaneval_data_10000.jsonl",
    "alpaca": "/data/tkumar/huggingface/processed_datasets/alpaca/alpaca_data_10000.jsonl",
    "c4": "/data/tkumar/huggingface/processed_datasets/c4/c4_data_10000.jsonl",
    "gsm": "/data/tkumar/huggingface/processed_datasets/gsm8k/gsm8k_data_10000.jsonl",
    "ultrafeedback": "/data/tkumar/huggingface/processed_datasets/ultrafeedback/ultrafeedback_data_10000.jsonl",
}


def _get_snapshot_path(base_path: str) -> str:
    """Resolve a model directory to an actual snapshot directory containing config.json.

    Accepts either:
    - a snapshot directory itself (contains config.json)
    - a "snapshots" parent dir (will pick the first child)
    - a base dir containing subdirs with config.json
    """
    if os.path.isdir(base_path):
        # Already a snapshot
        if os.path.exists(os.path.join(base_path, "config.json")):
            return base_path

        # Look for huggingface-style snapshots dir
        snapshots_dir = os.path.join(base_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            for item in os.listdir(snapshots_dir):
                item_path = os.path.join(snapshots_dir, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                    return item_path

        # Otherwise, try direct children
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                return item_path

    raise FileNotFoundError(
        f"No snapshot (config.json) found under {base_path}")


def _get_draft_model_path(args, cache_dir: str) -> str:
    """Get draft model path based on size or explicit directory."""
    if args.draft is not None and os.path.isdir(args.draft):
        return args.draft
    
    # Handle EAGLE auto-selection
    if getattr(args, "eagle", False):
        if args.llama:
            # Map target model size to corresponding EAGLE draft
            if args.size == "8":
                return os.path.join(cache_dir, "models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B")
            elif args.size == "70":
                return os.path.join(cache_dir, "models--yuhuili--EAGLE3-LLaMA3.3-Instruct-70B")
            else:
                raise ValueError(f"EAGLE draft not available for Llama size {args.size}")
        else:
            # Qwen
            if args.size == "32":
                return os.path.join(cache_dir, "models--RedHatAI--Qwen3-32B-speculator.eagle3")
            else:
                raise ValueError(f"EAGLE draft not available for Qwen size {args.size}")

    if args.llama:
        draft_size_to_model = {
            "1": "Llama-3.2-1B-Instruct",
            "3": "Llama-3.2-3B-Instruct",
            "8": "Llama-3.1-8B-Instruct",
            "70": "Llama-3.1-70B-Instruct",
        }
        if args.draft not in draft_size_to_model:
            raise ValueError(
                f"Draft size {args.draft} not available for Llama models. Available sizes: {list(draft_size_to_model.keys())}"
            )
        draft_model_name = draft_size_to_model[args.draft]
        return os.path.join(cache_dir, f"models--meta-llama--{draft_model_name}")
    else:
        draft_size_to_model = {
            "0.6": "Qwen3-0.6B",
            "1": "Llama-3.2-1B-Instruct",
        }
        if args.draft not in draft_size_to_model:
            raise ValueError(
                f"Draft size {args.draft} not available for Qwen models. Available sizes: {list(draft_size_to_model.keys())}"
            )
        draft_model_name = draft_size_to_model[args.draft]
        if args.draft == "1":
            return os.path.join(cache_dir, f"models--meta-llama--{draft_model_name}")
        else:
            return os.path.join(cache_dir, f"models--Qwen--{draft_model_name}")


def get_model_paths(args, cache_dir: str = "/data/tkumar/huggingface/hub/") -> Tuple[str, str, Optional[str]]:
    """Resolve model and draft paths (pointing to snapshot dirs with config.json)."""
    if args.llama:
        size_to_model = {
            "1": "Llama-3.2-1B-Instruct",
            "3": "Llama-3.2-3B-Instruct",
            "8": "Llama-3.1-8B-Instruct",
            "70": "Llama-3.1-70B-Instruct",
        }
        if args.size not in size_to_model:
            raise ValueError(
                f"Size {args.size} not available for Llama models. Available sizes: {list(size_to_model.keys())}"
            )
        model_name = size_to_model[args.size]
        model_base = os.path.join(
            cache_dir, f"models--meta-llama--{model_name}")
        default_draft_base = os.path.join(
            cache_dir, "models--meta-llama--Llama-3.2-1B-Instruct")
    else:
        size_to_model = {
            "0.6": "Qwen3-0.6B",
            "1.7": "Qwen3-1.7B",
            "4": "Qwen3-4B",
            "8": "Qwen3-8B",
            "14": "Qwen3-14B",
            "32": "Qwen3-32B",
        }
        if args.size not in size_to_model:
            raise ValueError(
                f"Size {args.size} not available for Qwen models. Available sizes: {list(size_to_model.keys())}"
            )
        model_name = size_to_model[args.size]
        model_base = os.path.join(cache_dir, f"models--Qwen--{model_name}")
        default_draft_base = os.path.join(
            cache_dir, "models--Qwen--Qwen3-0.6B")

    model_path = _get_snapshot_path(model_base)

    # Always resolve a draft path so callers can pass it through unchanged
    if getattr(args, "eagle", False) or getattr(args, "draft", None):
         draft_base = _get_draft_model_path(args, cache_dir)
    else:
         draft_base = default_draft_base
         
    draft_path = _get_snapshot_path(draft_base)

    return model_name, model_path, draft_path


def load_dataset_token_ids(
    dataset_name: str,
    model_path: str,
    num_prompts: int,
    input_len: int,
) -> Optional[List[List[int]]]:
    """Load and tokenize dataset prompts to token ids, padding/truncating to target length.

    target_len = max(len(text_tokens), input_len)
    """
    if dataset_name not in DATASET_PATHS:
        print(
            f"Warning: Unknown dataset {dataset_name}, falling back to random tokens")
        return None

    dataset_file_path = DATASET_PATHS[dataset_name]
    if not os.path.exists(dataset_file_path):
        print(
            f"Warning: Dataset file not found at {dataset_file_path}, falling back to random tokens")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        prompts: List[List[int]] = []
        with open(dataset_file_path, "r") as f:
            for _, line in enumerate(f):
                if len(prompts) >= num_prompts:
                    break
                data = json.loads(line.strip())
                text: str = data["text"]
                # tokens = tokenizer.encode(text, add_special_tokens=False)
                
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                tokens = tokenizer.encode(formatted, add_special_tokens=False)

                target_len = max(len(tokens), input_len)
                
                if len(tokens) >= target_len:
                    truncated_tokens = tokens[:target_len]
                else:
                    truncated_tokens = tokens
                prompts.append(truncated_tokens)
        return prompts
    except Exception as e:
        print(
            f"Warning: Error loading {dataset_name} prompts: {e}, falling back to random tokens")
        return None


def load_all_dataset_token_ids(
    model_path: str,
    num_prompts_per_dataset: int,
    input_len: int,
) -> List[List[int]]:
    """Load tokenized prompts from a union of datasets, falling back to random when needed."""
    datasets = ["humaneval", "alpaca", "gsm", "ultrafeedback"]
    all_prompts: List[List[int]] = []

    for dataset_name in datasets:
        print(
            f"Loading {num_prompts_per_dataset} prompts from {dataset_name}...")
        dataset_prompts = load_dataset_token_ids(
            dataset_name, model_path, num_prompts_per_dataset, input_len)
        if dataset_prompts is not None:
            all_prompts.extend(dataset_prompts)
        else:
            print(
                f"Failed to load {dataset_name}, adding random tokens instead")
            random_prompts = [[randint(0, 10000) for _ in range(
                input_len)] for _ in range(num_prompts_per_dataset)]
            all_prompts.extend(random_prompts)

    print(f"Total prompts loaded: {len(all_prompts)}")
    return all_prompts


def generate_benchmark_inputs(
    args,
    model_path: str,
) -> Tuple[Optional[List[str]], Optional[List[List[int]]], Optional[List[str]]]:
    """Create input prompts.

    Returns (string_prompts, prompt_token_ids, original_prompts)
    - string_prompts: list[str] when --example is used (chat template applied)
    - prompt_token_ids: list[list[int]] in dataset/random/all modes
    - original_prompts: for display when --example
    """
    if getattr(args, "example", False):
        example_prompts = [
            "introduce yourself",
            "explain the concept of recursion",
            "describe the color blue",
            "what are you doing?",
            "how do you feel?",
            "what's the weather like today?",
            "tell me a joke",
            "what is the meaning of life?",
        ]
        num_prompts = min(args.numseqs, len(example_prompts))
        selected_prompts = example_prompts[:num_prompts]

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        string_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in selected_prompts
        ]
        return string_prompts, None, selected_prompts

    if getattr(args, "random", False):
        prompt_token_ids = [[randint(0, 10000) for _ in range(
            args.input_len)] for _ in range(args.numseqs)]
        return None, prompt_token_ids, None

    if getattr(args, "all", False):
        token_ids = load_all_dataset_token_ids(
            model_path, args.numseqs, args.input_len)
        if not token_ids:
            print("Warning: All dataset loading failed, falling back to random tokens")
            token_ids = [[randint(0, 10000) for _ in range(
                args.input_len)] for _ in range(args.numseqs * 4)]
        return None, token_ids, None

    # Single dataset case
    if getattr(args, "humaneval", False):
        dataset_name = "humaneval"
    elif getattr(args, "alpaca", False):
        dataset_name = "alpaca"
    elif getattr(args, "c4", False):
        dataset_name = "c4"
    elif getattr(args, "ultrafeedback", False):
        dataset_name = "ultrafeedback"
    else:
        dataset_name = "gsm"

    dataset_prompts = load_dataset_token_ids(
        dataset_name, model_path, args.numseqs, args.input_len)
    if dataset_prompts is None:
        token_ids = [[randint(0, 10000) for _ in range(args.input_len)]
                     for _ in range(args.numseqs)]
    else:
        token_ids = dataset_prompts
    return None, token_ids, None
