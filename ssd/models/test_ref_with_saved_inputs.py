#!/usr/bin/env python3
"""
Test script that loads saved draft prefill inputs and runs them through the ref model.
This helps debug the EAGLE3 draft implementation by comparing against the reference.

Usage:
    # First, run bench.py with --debug to save inputs:
    python bench/bench.py --llama --size 8 --eagle --jit --eager --debug --numseqs 1 --b 1
    
    # Then run this script to test with ref model (uses same defaults as main pipeline):
    python ssd/models/test_ref_with_saved_inputs.py
    
    # Or specify custom paths:
    python ssd/models/test_ref_with_saved_inputs.py \
        --model-path /path/to/eagle3/draft/model \
        --target-path /path/to/target/model
"""
import os
import sys
import argparse
import torch
from transformers import AutoConfig, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssd.models.ref import Model as RefModel, EagleConfig
from ssd.utils.loader import default_weight_loader


def load_eagle_weights(model, model_path, target_path=None):
    """Load EAGLE3 weights into ref model (similar to compare.py)"""
    print(f"Loading EAGLE3 weights from {model_path}")
    bin_file = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"Expected pytorch_model.bin at {bin_file}")
    
    state_dict = torch.load(bin_file, map_location="cpu")
    
    # Check for embedding layer
    found_embed_tokens = False
    for weight_name in state_dict.keys():
        if 'embed_tokens' in weight_name:
            found_embed_tokens = True
            break
    
    # Load embeddings from target if not in draft weights
    if not found_embed_tokens and target_path:
        print(f"'embed_tokens' not found in draft weights, loading from target: {target_path}")
        if not load_embeddings_from_target(model, target_path):
            raise ValueError("Failed to load embeddings from target model")
    
    # Load weights into ref model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    with torch.no_grad():
        for weight_name, loaded_weight in state_dict.items():
            if weight_name in ['d2t', 't2d']:
                continue
            
            loaded_weight = loaded_weight.to(device=device, dtype=dtype)
            
            if weight_name == 'midlayer.hidden_norm.weight':
                model.midlayer.hidden_norm.weight.data.copy_(loaded_weight)
            elif weight_name.startswith('midlayer.'):
                param_name = weight_name.replace('midlayer.', '')
                parts = param_name.split('.')
                obj = model.midlayer
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                getattr(obj, parts[-1]).data.copy_(loaded_weight)
            elif weight_name == 'fc.weight':
                model.fc.weight.data.copy_(loaded_weight)
            elif weight_name == 'norm.weight':
                model.norm.weight.data.copy_(loaded_weight)
            elif weight_name == 'lm_head.weight':
                model.lm_head.weight.data.copy_(loaded_weight)
            elif weight_name == 'embed_tokens.weight':
                model.embed_tokens.weight.data.copy_(loaded_weight)
    
    print("Finished loading weights")


def load_embeddings_from_target(model, target_path):
    """Load embedding weights from target model"""
    import glob
    target_keys = ["model.embed_tokens.weight", "embed_tokens.weight"]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # Try safetensors first
    safetensor_files = glob.glob(os.path.join(target_path, "*.safetensors"))
    for file in safetensor_files:
        try:
            from safetensors import safe_open
            with safe_open(file, "pt", "cpu") as f:
                keys = f.keys()
                for key in target_keys:
                    if key in keys:
                        print(f"Found embedding {key} in {file}")
                        tensor = f.get_tensor(key)
                        tensor_device = tensor.to(device=device, dtype=dtype)
                        with torch.no_grad():
                            model.embed_tokens.weight.data.copy_(tensor_device)
                        return True
        except Exception as e:
            print(f"Error reading safetensor {file}: {e}")
            continue
    
    # Try bin files
    bin_files = glob.glob(os.path.join(target_path, "pytorch_model*.bin"))
    for file in bin_files:
        try:
            print(f"Checking {file} for embeddings...")
            state_dict = torch.load(file, map_location="cpu")
            for key in target_keys:
                if key in state_dict:
                    print(f"Found embedding {key} in {file}")
                    tensor = state_dict[key]
                    tensor_device = tensor.to(device=device, dtype=dtype)
                    with torch.no_grad():
                        model.embed_tokens.weight.data.copy_(tensor_device)
                    return True
        except Exception as e:
            print(f"Error reading bin {file}: {e}")
            continue
    
    return False


def _get_snapshot_path(base_path: str) -> str:
    """Resolve a model directory to an actual snapshot directory containing config.json."""
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

    raise FileNotFoundError(f"No snapshot (config.json) found under {base_path}")


def main():
    # Default paths matching main pipeline (bench.py with --eagle --llama --size 8)
    cache_dir = "/data/tkumar/huggingface/hub/"
    default_target_base = os.path.join(cache_dir, "models--meta-llama--Llama-3.1-8B-Instruct")
    default_draft_base = os.path.join(cache_dir, "models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B")
    
    parser = argparse.ArgumentParser(description="Test ref model with saved draft inputs")
    parser.add_argument("--input-file", type=str, default="debug_outputs/draft_prefill_inputs.pt",
                        help="Path to saved draft inputs")
    parser.add_argument("--model-path", type=str, default=None,
                        help=f"Path to EAGLE3 draft model weights (default: {default_draft_base})")
    parser.add_argument("--target-path", type=str, default=None,
                        help=f"Path to target model (default: {default_target_base})")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type (float32, float16, bfloat16)")
    args = parser.parse_args()
    
    # Use defaults if not provided
    if args.model_path is None:
        args.model_path = default_draft_base
        print(f"Using default draft model path: {args.model_path}")
    
    if args.target_path is None:
        args.target_path = default_target_base
        print(f"Using default target model path: {args.target_path}")
    device = torch.device(args.device)
    
    # Resolve to snapshot paths
    try:
        args.model_path = _get_snapshot_path(args.model_path)
        args.target_path = _get_snapshot_path(args.target_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        print("Make sure you've run bench.py with --eagle --jit --eager --debug first")
        sys.exit(1)
    
    # Load saved inputs
    print(f"\nLoading saved inputs from {args.input_file}")
    debug_data = torch.load(args.input_file, map_location="cpu")
    
    input_ids = debug_data['input_ids']
    positions = debug_data['positions']
    target_hidden_states = debug_data['target_hidden_states']
    token_embeddings = debug_data['token_embeddings']
    d_model_target = debug_data['d_model_target']
    eagle_layers = debug_data['eagle_layers']
    
    print(f"\nLoaded inputs:")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  positions shape: {positions.shape}")
    print(f"  target_hidden_states shape: {target_hidden_states.shape}")
    print(f"  token_embeddings shape: {token_embeddings.shape}")
    print(f"  d_model_target: {d_model_target}")
    print(f"  eagle_layers: {eagle_layers}")
    
    # Infer config from model path
    print(f"\nInferring config from {args.model_path}")
    draft_hf_config = AutoConfig.from_pretrained(args.model_path)
    
    # Get target config
    print(f"Loading target config from {args.target_path}")
    target_hf_config = AutoConfig.from_pretrained(args.target_path)
    target_hidden_size = target_hf_config.hidden_size
    
    # Load vocab sizes and d2t mapping from weights if available
    bin_file = os.path.join(args.model_path, "pytorch_model.bin")
    d2t_tensor = None
    if os.path.exists(bin_file):
        state_dict = torch.load(bin_file, map_location="cpu")
        if 'd2t' in state_dict and 't2d' in state_dict:
            target_vocab_size = len(state_dict['t2d'])
            draft_vocab_size = len(state_dict['d2t'])
            d2t_tensor = state_dict['d2t'].to(device)
            print(f"Detected vocab sizes from weights: target={target_vocab_size}, draft={draft_vocab_size}")
            print(f"Loaded d2t mapping: {d2t_tensor.shape}")
        else:
            target_vocab_size = draft_hf_config.vocab_size
            draft_vocab_size = draft_hf_config.vocab_size
    else:
        target_vocab_size = draft_hf_config.vocab_size
        draft_vocab_size = draft_hf_config.vocab_size
    
    # Create ref config
    ref_config = EagleConfig(
        hidden_size=draft_hf_config.hidden_size,
        intermediate_size=draft_hf_config.intermediate_size,
        num_attention_heads=draft_hf_config.num_attention_heads,
        num_key_value_heads=draft_hf_config.num_key_value_heads,
        vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        max_position_embeddings=draft_hf_config.max_position_embeddings,
        rms_norm_eps=draft_hf_config.rms_norm_eps,
        rope_theta=getattr(draft_hf_config, 'rope_theta', 10000.0),
        target_hidden_size=target_hidden_size,
        pad_token_id=0,
        hidden_act="silu"
    )
    
    # Setup device and dtype
    
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    
    print(f"\nCreating ref model on {device} with {dtype}")
    ref_model = RefModel(ref_config).to(device=device, dtype=dtype)
    ref_model.eval()
    
    # Load weights
    load_eagle_weights(ref_model, args.model_path, args.target_path)
    
    # Load tokenizer for decoding
    print(f"\nLoading tokenizer from {args.target_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_path)
    
    # Print detokenized input prompt
    print("\n" + "="*80)
    print("INPUT PROMPT (DETOKENIZED):")
    print("="*80)
    
    # Decode the full input_ids to show the prompt
    if input_ids.dim() == 1:
        # Single flattened sequence
        decoded_prompt = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
        print(f"\nFull prompt ({len(input_ids)} tokens):")
        print(repr(decoded_prompt))
    else:
        # Multiple sequences
        for seq_idx in range(input_ids.shape[0]):
            seq_tokens = input_ids[seq_idx].tolist()
            decoded_prompt = tokenizer.decode(seq_tokens, skip_special_tokens=False)
            print(f"\nSequence {seq_idx} ({len(seq_tokens)} tokens):")
            print(repr(decoded_prompt))
    
    # Move inputs to device
    input_ids = input_ids.to(device)
    positions = positions.to(device)
    target_hidden_states = target_hidden_states.to(device=device, dtype=dtype)
    
    # Reshape to 2D for ref model (assumes batch size 1 or flattened)
    # Detect batch size by checking if positions reset
    num_tokens = input_ids.shape[0]
    if positions.dim() == 1:
        # Find where positions reset to detect sequences
        pos_diffs = positions[1:] - positions[:-1]
        seq_boundaries = torch.where(pos_diffs <= 0)[0] + 1
        if len(seq_boundaries) > 0:
            # Multiple sequences
            bs = len(seq_boundaries) + 1
            seq_lens = torch.diff(torch.cat([torch.tensor([0], device=device), 
                                             seq_boundaries, 
                                             torch.tensor([num_tokens], device=device)]))
            seq_len = seq_lens[0].item()  # Assume all same length
            
            input_ids_2d = input_ids.view(bs, seq_len)
            positions_2d = positions.view(bs, seq_len)
            target_hidden_states_2d = target_hidden_states.view(bs, seq_len, -1)
        else:
            # Single sequence
            bs = 1
            seq_len = num_tokens
            input_ids_2d = input_ids.unsqueeze(0)
            positions_2d = positions.unsqueeze(0)
            target_hidden_states_2d = target_hidden_states.unsqueeze(0)
    else:
        # Already 2D
        input_ids_2d = input_ids
        positions_2d = positions
        bs, seq_len = input_ids_2d.shape
        target_hidden_states_2d = target_hidden_states.view(bs, seq_len, -1)
    
    print(f"\nReshaped for ref model (bs={bs}, seq_len={seq_len}):")
    print(f"  input_ids_2d: {input_ids_2d.shape}")
    print(f"  positions_2d: {positions_2d.shape}")
    target_hidden_states_2d = torch.zeros_like(target_hidden_states_2d) 
    print(f"  target_hidden_states_2d: {target_hidden_states_2d.shape}")
    
    # Run forward pass
    print("\n" + "="*80)
    print("Running forward pass through ref model...")
    print("="*80)
    
    with torch.no_grad():
        hidden_states = ref_model(
            target_hidden_states_2d, 
            input_ids_2d, 
            position_ids=positions_2d.long(), 
            use_cache=False
        )
        
        # Compute logits
        normed = ref_model.norm(hidden_states)
        logits = ref_model.lm_head(normed)
    
    print(f"\nOutput shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  normed: {normed.shape}")
    print(f"  logits: {logits.shape}")
    
    # Get last token logits for each sequence
    print("\n" + "="*80)
    print("TOP 5 LOGITS AT LAST TOKEN POSITION:")
    print("="*80)
    
    for seq_idx in range(bs):
        last_token_logits = logits[seq_idx, -1, :]  # [draft_vocab_size]
        top5_values, top5_indices = torch.topk(last_token_logits, k=5)
        
        print(f"\nSequence {seq_idx}:")
        print(f"  Last token input_id: {input_ids_2d[seq_idx, -1].item()}")
        print(f"  Last token position: {positions_2d[seq_idx, -1].item()}")
        print(f"  Top 5 predictions:")
        
        for rank, (draft_token_id, logit_value) in enumerate(zip(top5_indices, top5_values)):
            draft_id = draft_token_id.item()
            
            # Apply d2t mapping: target_token_id = draft_token_id + d2t[draft_token_id]
            if d2t_tensor is not None:
                target_id = draft_id + d2t_tensor[draft_id].item()
            else:
                target_id = draft_id  # Identity mapping if no d2t
            
            # Decode the target token
            try:
                token_text = tokenizer.decode([target_id])
            except:
                token_text = "<decode_error>"
            
            print(f"    {rank+1}. draft_id={draft_id:5d} -> target_id={target_id:6d}, logit={logit_value.item():8.3f}, text={repr(token_text)}")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()

