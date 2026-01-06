#!/usr/bin/env python3
"""Benchmark EAGLE3 draft model forward pass timing."""
import os
import sys
import argparse
import time
import torch
from transformers import AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssd.models.ref import Model as RefModel, EagleConfig


def _get_snapshot_path(base_path: str) -> str:
    if os.path.isdir(base_path):
        if os.path.exists(os.path.join(base_path, "config.json")):
            return base_path
        snapshots_dir = os.path.join(base_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            for item in os.listdir(snapshots_dir):
                item_path = os.path.join(snapshots_dir, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                    return item_path
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                return item_path
    raise FileNotFoundError(f"No snapshot (config.json) found under {base_path}")


def load_embeddings_from_target(model, target_path, device, dtype):
    import glob
    from safetensors import safe_open
    target_keys = ["model.embed_tokens.weight", "embed_tokens.weight"]
    
    for file in glob.glob(os.path.join(target_path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for key in target_keys:
                if key in f.keys():
                    tensor = f.get_tensor(key).to(device=device, dtype=dtype)
                    model.embed_tokens.weight.data.copy_(tensor)
                    return True
    return False


def load_eagle_weights(model, model_path, target_path, device, dtype):
    bin_file = os.path.join(model_path, "pytorch_model.bin")
    state_dict = torch.load(bin_file, map_location="cpu")
    
    found_embed = any('embed_tokens' in k for k in state_dict.keys())
    if not found_embed and target_path:
        load_embeddings_from_target(model, target_path, device, dtype)
    
    with torch.no_grad():
        for name, weight in state_dict.items():
            if name in ['d2t', 't2d']:
                continue
            weight = weight.to(device=device, dtype=dtype)
            if name == 'midlayer.hidden_norm.weight':
                model.midlayer.hidden_norm.weight.data.copy_(weight)
            elif name.startswith('midlayer.'):
                parts = name.replace('midlayer.', '').split('.')
                obj = model.midlayer
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                getattr(obj, parts[-1]).data.copy_(weight)
            elif name == 'fc.weight':
                model.fc.weight.data.copy_(weight)
            elif name == 'norm.weight':
                model.norm.weight.data.copy_(weight)
            elif name == 'lm_head.weight':
                model.lm_head.weight.data.copy_(weight)
            elif name == 'embed_tokens.weight':
                model.embed_tokens.weight.data.copy_(weight)


@torch.inference_mode()
def capture_prefill_cudagraph(model, bs, seq_len, hidden_dim, vocab_size, device, dtype):
    """Capture CUDA graph for prefill following cudagraph_helpers.py pattern."""
    # Static buffers
    input_ids = torch.zeros(bs, seq_len, dtype=torch.int64, device=device)
    positions = torch.zeros(bs, seq_len, dtype=torch.int64, device=device)
    hidden_states = torch.zeros(bs, seq_len, hidden_dim, dtype=dtype, device=device)
    outputs = torch.zeros(bs, seq_len, model.hidden_size, dtype=dtype, device=device)
    
    # Warmup
    outputs[:] = model(hidden_states, input_ids, position_ids=positions, use_cache=False)
    torch.cuda.synchronize()
    
    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs[:] = model(hidden_states, input_ids, position_ids=positions, use_cache=False)
    torch.cuda.synchronize()
    
    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        hidden_states=hidden_states,
        outputs=outputs,
    )
    return graph, graph_vars


@torch.inference_mode()
def capture_decode_cudagraph(model, bs, seq_len, hidden_dim, vocab_size, device, dtype):
    """Capture CUDA graph for decode (1 token with KV cache) following cudagraph_helpers.py pattern."""
    # Static buffers for prefill (to get KV cache)
    prefill_ids = torch.zeros(bs, seq_len, dtype=torch.int64, device=device)
    prefill_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1).contiguous()
    prefill_hidden = torch.zeros(bs, seq_len, hidden_dim, dtype=dtype, device=device)
    
    # Run prefill to get KV cache structure
    _, kv = model(prefill_hidden, prefill_ids, position_ids=prefill_pos, use_cache=True)
    torch.cuda.synchronize()
    
    # Static buffers for decode
    decode_ids = torch.zeros(bs, 1, dtype=torch.int64, device=device)
    decode_pos = torch.full((bs, 1), seq_len, dtype=torch.int64, device=device)
    decode_hidden = torch.zeros(bs, 1, hidden_dim, dtype=dtype, device=device)
    outputs = torch.zeros(bs, 1, model.hidden_size, dtype=dtype, device=device)
    
    # Clone KV cache to static buffers
    kv_static = tuple((k.clone(), v.clone()) for k, v in kv)
    
    # Warmup decode
    out, _ = model(decode_hidden, decode_ids, position_ids=decode_pos, past_key_values=kv_static, use_cache=True)
    outputs[:] = out
    torch.cuda.synchronize()
    
    # Re-clone KV for capture (warmup modified it)
    kv_static = tuple((k.clone(), v.clone()) for k, v in kv)
    
    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, _ = model(decode_hidden, decode_ids, position_ids=decode_pos, past_key_values=kv_static, use_cache=True)
        outputs[:] = out
    torch.cuda.synchronize()
    
    graph_vars = dict(
        decode_ids=decode_ids,
        decode_pos=decode_pos,
        decode_hidden=decode_hidden,
        outputs=outputs,
    )
    return graph, graph_vars, kv_static


def main():
    cache_dir = "/data/tkumar/huggingface/hub/"
    default_target = os.path.join(cache_dir, "models--meta-llama--Llama-3.1-8B-Instruct")
    default_draft = os.path.join(cache_dir, "models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=default_draft)
    parser.add_argument("--target-path", default=default_target)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--prefill", action="store_true", help="Bench prefill instead of decode")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--cudagraph", action="store_true", help="Use CUDA graphs")
    parser.add_argument("--check", action="store_true", help="Verify cudagraph correctness vs eager")
    args = parser.parse_args()
    
    if args.compile and args.cudagraph:
        print("Warning: --compile and --cudagraph are mutually exclusive. Using --cudagraph only.")
        args.compile = False
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    model_path = _get_snapshot_path(args.model_path)
    target_path = _get_snapshot_path(args.target_path)
    
    draft_cfg = AutoConfig.from_pretrained(model_path)
    target_cfg = AutoConfig.from_pretrained(target_path)
    
    bin_file = os.path.join(model_path, "pytorch_model.bin")
    state_dict = torch.load(bin_file, map_location="cpu")
    target_vocab = len(state_dict['t2d']) if 't2d' in state_dict else draft_cfg.vocab_size
    draft_vocab = len(state_dict['d2t']) if 'd2t' in state_dict else draft_cfg.vocab_size
    
    config = EagleConfig(
        hidden_size=draft_cfg.hidden_size,
        intermediate_size=draft_cfg.intermediate_size,
        num_attention_heads=draft_cfg.num_attention_heads,
        num_key_value_heads=draft_cfg.num_key_value_heads,
        vocab_size=target_vocab,
        draft_vocab_size=draft_vocab,
        max_position_embeddings=draft_cfg.max_position_embeddings,
        rms_norm_eps=draft_cfg.rms_norm_eps,
        rope_theta=getattr(draft_cfg, 'rope_theta', 10000.0),
        target_hidden_size=target_cfg.hidden_size,
        pad_token_id=0,
        hidden_act="silu"
    )
    
    print(f"Creating model: hidden={config.hidden_size}, target_hidden={config.target_hidden_size}")
    model = RefModel(config).to(device=device, dtype=dtype)
    model.eval()
    
    print("Loading weights...")
    load_eagle_weights(model, model_path, target_path, device, dtype)
    
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    bs, seq_len = args.bs, args.seq_len
    hidden_dim = config.target_hidden_size * 3
    
    if args.prefill:
        print(f"\nBenchmarking prefill (bs={bs}, seq_len={seq_len})", end="")
        if args.cudagraph:
            print(" [CUDA graph]")
            graph, graph_vars = capture_prefill_cudagraph(
                model, bs, seq_len, hidden_dim, target_vocab, device, dtype)
            
            # Buffers already set during capture, no need to fill for benchmark
            
            # Warmup
            for _ in range(args.warmup):
                graph.replay()
            torch.cuda.synchronize()
            
            # Timed
            times = []
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        else:
            print()
            input_ids = torch.randint(0, target_vocab, (bs, seq_len), device=device)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
            hidden_states = torch.randn(bs, seq_len, hidden_dim, device=device, dtype=dtype)
            
            for _ in range(args.warmup):
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    model(hidden_states, input_ids, position_ids=positions, use_cache=False)
            torch.cuda.synchronize()
            
            times = []
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    model(hidden_states, input_ids, position_ids=positions, use_cache=False)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
    else:
        print(f"\nBenchmarking decode (bs={bs}, ctx={seq_len})", end="")
        if args.cudagraph:
            print(" [CUDA graph]")
            graph, graph_vars, _ = capture_decode_cudagraph(
                model, bs, seq_len, hidden_dim, target_vocab, device, dtype)
            
            # Buffers already have zeros, no need to fill for benchmark
            
            # Warmup
            for _ in range(args.warmup):
                graph.replay()
            torch.cuda.synchronize()
            
            # Timed
            times = []
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        else:
            print()
            input_ids = torch.randint(0, target_vocab, (bs, seq_len), device=device)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
            hidden_states = torch.randn(bs, seq_len, hidden_dim, device=device, dtype=dtype)
            
            with torch.no_grad():
                _, kv = model(hidden_states, input_ids, position_ids=positions, use_cache=True)
            
            decode_ids = torch.randint(0, target_vocab, (bs, 1), device=device)
            decode_pos = torch.full((bs, 1), seq_len, device=device, dtype=torch.long)
            decode_hidden = torch.randn(bs, 1, hidden_dim, device=device, dtype=dtype)
            
            for _ in range(args.warmup):
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    model(decode_hidden, decode_ids, position_ids=decode_pos, past_key_values=kv, use_cache=True)
            torch.cuda.synchronize()
            
            times = []
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    model(decode_hidden, decode_ids, position_ids=decode_pos, past_key_values=kv, use_cache=True)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
    
    times = torch.tensor(times) * 1000  # ms
    print(f"\nResults ({args.iters} iters):")
    print(f"  Mean: {times.mean():.3f} ms")
    print(f"  Std:  {times.std():.3f} ms")
    print(f"  Min:  {times.min():.3f} ms")
    print(f"  Max:  {times.max():.3f} ms")
    
    # Correctness check
    if args.check and args.cudagraph:
        print("\nVerifying correctness...")
        if args.prefill:
            # Run eager with same inputs
            with torch.no_grad():
                eager_out = model(graph_vars["hidden_states"], graph_vars["input_ids"], 
                                  position_ids=graph_vars["positions"], use_cache=False)
            graph.replay()
            torch.cuda.synchronize()
            graph_out = graph_vars["outputs"].clone()
            
            max_diff = (eager_out - graph_out).abs().max().item()
            mean_diff = (eager_out - graph_out).abs().mean().item()
            print(f"  Eager vs Graph - max diff: {max_diff:.6e}, mean diff: {mean_diff:.6e}")
            print(f"  Output norm: {graph_out.norm().item():.4f}")
            if max_diff < 1e-3:
                print("  ✓ PASSED")
            else:
                print("  ✗ FAILED - outputs differ significantly")
        else:
            # For decode, compare graph output with eager
            # Re-run prefill to get fresh KV cache for eager
            prefill_ids = torch.zeros(bs, seq_len, dtype=torch.int64, device=device)
            prefill_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
            prefill_hidden = torch.zeros(bs, seq_len, hidden_dim, dtype=dtype, device=device)
            
            with torch.no_grad():
                _, kv_eager = model(prefill_hidden, prefill_ids, position_ids=prefill_pos, use_cache=True)
            
            decode_ids = graph_vars["decode_ids"].clone()
            decode_pos = graph_vars["decode_pos"].clone()
            decode_hidden = graph_vars["decode_hidden"].clone()
            
            with torch.no_grad():
                eager_out, _ = model(decode_hidden, decode_ids, position_ids=decode_pos, 
                                     past_key_values=kv_eager, use_cache=True)
            
            graph.replay()
            torch.cuda.synchronize()
            graph_out = graph_vars["outputs"].clone()
            
            max_diff = (eager_out - graph_out).abs().max().item()
            mean_diff = (eager_out - graph_out).abs().mean().item()
            print(f"  Eager vs Graph - max diff: {max_diff:.6e}, mean diff: {mean_diff:.6e}")
            print(f"  Output norm: {graph_out.norm().item():.4f}")
            if max_diff < 1e-3:
                print("  ✓ PASSED")
            else:
                print("  ✗ FAILED - outputs differ significantly")


if __name__ == "__main__":
    main()
