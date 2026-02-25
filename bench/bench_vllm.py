import os
import time
import argparse
from random import seed
from typing import Tuple
import sys 
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import wandb
from bench_helpers import get_model_paths, generate_benchmark_inputs


## model paths now provided by bench_helpers.get_model_paths


## dataset helpers now provided by bench_helpers


## data generation now from bench_helpers.generate_benchmark_inputs

def main():
    parser = argparse.ArgumentParser(description="Benchmark real vLLM with Qwen3/Llama3 models and optional speculative decoding.")
    
    # Model and infrastructure
    parser.add_argument("--size", type=str, choices=["0.6", "1.7", "4", "8", "14", "32", "1", "3", "70"], default="0.6",
                        help="Model size in billions of parameters")
    parser.add_argument("--gpus", type=int, default=1, help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--llama", action="store_true", help="Use Llama models instead of Qwen")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Override max model len")
    
    # Speculative decoding
    parser.add_argument("--spec", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--eagle", action="store_true", help="Enable EAGLE3 speculative decoding")
    parser.add_argument("--k", type=int, default=1, help="Number of speculative tokens (lookahead)")
    parser.add_argument("--draft", type=str, default=None,
                        help="Draft model size (e.g., 1 for Llama-1B, 0.6 for Qwen-0.6B) or full path to draft model")
    
    # Generation parameters
    parser.add_argument("--input_len", type=int, default=128, help="Prompt length in tokens")
    parser.add_argument("--output_len", type=int, default=512, help="New tokens to generate per request")
    parser.add_argument("--numseqs", type=int, default=1, help="Number of requests to generate (batch size)")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature for generation")
    
    # Other parameters  
    parser.add_argument("--b", type=int, default=1, help="Max in-flight sequences")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90, help="vLLM gpu_memory_utilization (0-1)")
    
    # Dataset selection
    parser.add_argument("--example", action="store_true", help="Use real prompts like in example.py and print generations")
    parser.add_argument("--humaneval", action="store_true", help="Use HumanEval prompts")
    parser.add_argument("--alpaca", action="store_true", help="Use Alpaca prompts")
    parser.add_argument("--c4", action="store_true", help="Use C4 prompts")
    parser.add_argument("--ultrafeedback", action="store_true", help="Use UltraFeedback prompts")
    parser.add_argument("--random", action="store_true", help="Use random tokens instead of dataset prompts")
    parser.add_argument("--all", action="store_true", help="Use numseqs from each dataset (union dataset with numseqs*4 total)")
    
    # Debugging and logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to wandb")
    parser.add_argument("--group", type=str, default=None, help="Wandb group name")
    parser.add_argument("--name", type=str, default=None, help="Wandb run name")
    
    args = parser.parse_args()

    seed(0)
    
    # Validate example mode constraints
    if args.example and args.numseqs > 8:
        print("Warning: --example mode supports up to 8 sequences, reducing numseqs from {} to 8".format(args.numseqs))
        args.numseqs = 8

    # Get model paths
    model_name, model_path, draft_path = get_model_paths(args)
    
    # Prepare inputs
    string_prompts, prompt_token_ids, original_prompts = generate_benchmark_inputs(args, model_path)

    # Create run name
    spec_mode_str = "eagle" if args.eagle else ("spec" if args.spec else "normal")
    model_type = "llama" if args.llama else "qwen"
    run_name = args.name if args.name else f"{model_type}_size_{args.size}_{spec_mode_str}_b{args.b}_k{args.k}_temp{args.temp}"

    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project="ssd",
            name=run_name,
            group=args.group,
            config={
                "model_size": args.size,
                "gpus": args.gpus,
                "speculative_decoding": args.spec,
                "eagle": args.eagle,
                "k": args.k if (args.spec or args.eagle) else None,
                "llama": args.llama,
                "max_model_len": args.max_model_len,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "numseqs": args.numseqs,
                "temperature": args.temp,
                "gpu_mem_util": args.gpu_mem_util,
                "draft_model": args.draft,
                "b": args.b,
                "example_mode": args.example,
                "humaneval_mode": args.humaneval,
                "alpaca_mode": args.alpaca,
                "c4_mode": args.c4,
                "ultrafeedback_mode": args.ultrafeedback,
                "random_mode": args.random,
                "all_mode": args.all,
                "implementation": "vllm",
            }
        )

    # Build LLM kwargs for vLLM
    llm_kwargs = dict(
        tensor_parallel_size=args.gpus,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.b,
        max_num_batched_tokens=8192, 
        generation_config="vllm",  # Ensure vanilla defaults, do not apply HF generation_config
        enforce_eager=False, 
    )

    # Speculative decoding configuration
    if args.eagle:
        print(f"Using EAGLE3 for speculative decoding with model {args.size}")
        if args.size == "70":
            eagle_model = "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"
        elif args.size == "8":
            assert args.llama, "EAGLE-8B only supports LLaMA models"
            eagle_model = "yuhuili/EAGLE-LLaMA3-Instruct-8B"
        else:
            raise ValueError(f"EAGLE not supported for model type/size {args.size}")
        
        speculative_config = dict(
            model=eagle_model,
            num_speculative_tokens=max(1, int(args.k)),
            method="eagle3",
            draft_tensor_parallel_size=1,
        )
        llm_kwargs.update(speculative_config=speculative_config)
    elif args.spec:
        if draft_path is None:
            raise ValueError("Speculative decoding requested but no draft model provided or resolved.")
        
        speculative_config = dict(
            model=draft_path,
            num_speculative_tokens=max(1, int(args.k)),
        )
        llm_kwargs.update(speculative_config=speculative_config)

    # Initialize vLLM
    llm = LLM(model=model_path, **llm_kwargs)

    # Determine total sequences
    if string_prompts is not None:
        num_reqs = len(string_prompts)
    elif prompt_token_ids is not None:
        num_reqs = len(prompt_token_ids)
    else:
        num_reqs = args.numseqs

    # Log initial progress to wandb
    if args.wandb:
        wandb.log({"sequences_processed": 0, "total_sequences": num_reqs})

    sampling_params = [SamplingParams(
        temperature=args.temp,
        max_tokens=args.output_len,
        ignore_eos=True,
        top_p=1.0,
        top_k=-1,
    ) for _ in range(num_reqs)]
    
    t0 = time.time()
    if prompt_token_ids is not None:
        outputs = llm.generate(prompts=None, prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    else:
        outputs = llm.generate(prompts=string_prompts, sampling_params=sampling_params)
    dt = time.time() - t0

    # Log completion to wandb
    if args.wandb:
        wandb.log({"sequences_processed": num_reqs, "total_sequences": num_reqs})

    # Compute throughput on generated tokens only
    total_new_tokens = 0
    for out in outputs:
        for seq in out.outputs:
            total_new_tokens += len(seq.token_ids)

    throughput = total_new_tokens / dt if dt > 0 else 0.0
    spec_mode = f" + EAGLE3(k={args.k})" if args.eagle else (f" + Speculative(k={args.k})" if args.spec else "")
    print(f"Model: {model_name}, Engine: vLLM{spec_mode}, Requests: {num_reqs}, New Tokens: {total_new_tokens}, Time: {dt:.2f}s, Throughput: {throughput:.2f} tok/s")

    # Print generations unless in random mode
    if not args.random:
        print("\n" + "="*80)
        print("GENERATIONS:")
        print("="*80)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Print at most 10 generations
        for i, output in enumerate(outputs):
            if i >= 10:
                break
                
            # Decode prompt if it's tokenized
            if prompt_token_ids is not None and i < len(prompt_token_ids):
                decoded_prompt = tokenizer.decode(prompt_token_ids[i], skip_special_tokens=True)
            else:
                decoded_prompt = string_prompts[i] if string_prompts and string_prompts[i] else ""
            
            # Use original prompt if available, otherwise decoded prompt
            if original_prompts and i < len(original_prompts):
                display_prompt = original_prompts[i]
            else:
                display_prompt = decoded_prompt
            
            print(f"\nPrompt {i+1}: {display_prompt!r}")
            print(f"Generation: {output.outputs[0].text!r}")
            print("-" * 40)

    # Log final metrics to wandb
    if args.wandb:
        wandb.log({
            "total_new_tokens": total_new_tokens,
            "time_seconds": dt,
            "throughput_tok_per_sec": throughput,
            "model_name": model_name,
            "speculative_mode": spec_mode,
            "run_name": run_name,
        })
        wandb.finish()

    sys.exit(0)

if __name__ == "__main__":
    main()