import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'  # for FlashInfer
import sys
import time
import argparse
import json
from random import randint, seed
from ssd import LLM, SamplingParams
from transformers import AutoTokenizer
import wandb
from bench_helpers import get_model_paths, generate_benchmark_inputs


def parse_arguments():
    """Parse command line arguments for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark SSD performance (API similar to example.py)")
    
    # Model configuration
    parser.add_argument("--size", type=str, choices=["0.6", "1.7", "4", "8", "14", "32", "1", "3", "70"], default="4", 
                        help="Model size in billions of parameters (0.6, 1.7, 4, 8, 14, 32, 1, 3, 70)")
    parser.add_argument("--llama", action="store_true", help="Use Llama models instead of Qwen")
    parser.add_argument("--draft", type=str, default=None, 
                        help="Draft model size (0.6 for Qwen-0.6B, 1 for Llama-1B) or path to draft model")
    
    # Execution configuration
    parser.add_argument("--eager", action="store_true", help="Use eager execution (disable CUDA graphs)")
    parser.add_argument("--gpus", type=int, default=1, help="Total number of gpus")
    
    # Speculative decoding configuration
    parser.add_argument("--spec", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--eagle", action="store_true", help="Enable eagle speculative decoding (implies --spec, uses default eagle draft for model)")
    parser.add_argument("--k", type=int, default=6, help="Speculative decoding k value")
    parser.add_argument("--async", action="store_true", help="Enable async speculative decoding")
    parser.add_argument("--f", type=int, default=3, help="Async fan out value")
    parser.add_argument("--fl", type=int, nargs='+', default=None, help="Fan out list (e.g., --fl 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--flh", type=int, nargs='+', default=None, help="Fan out list (e.g., --flh 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--flm", type=int, nargs='+', default=None, help="Fan out list miss (e.g., --flm 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--dtemp", type=float, default=None, help="Draft async temperature (overrides --temp for async tree decode)")
    parser.add_argument("--ttemp", type=float, default=None, help="Target async temperature (overrides --temp for async verify)")
    parser.add_argument("--afn", dest="afn", action="store_true", help="Enable adaptive fan-out (skip top-1 for 0<k<K)")
    parser.set_defaults(afn=False) # warning: do not use `afn` it is deprecated 
    parser.add_argument("--backup", type=str, choices=["jit", "fast"], default="jit", help="Backup strategy (jit or fast)")
    
    # Memory and batching configuration
    parser.add_argument("--block_sz", type=int, default=256, help="KV cache block size (see config.py: kvcache_block_size)")
    parser.add_argument("--b", type=int, default=1, help="Maximum number of sequences in batch")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum model length") # changed, see if this affects perf
    
    # Generation configuration
    parser.add_argument("--input_len", type=int, default=128, help="Maximum input length")
    parser.add_argument("--output_len", type=int, default=512, help="Maximum output length")
    parser.add_argument("--numseqs", type=int, default=128, help="Number of sequences to generate")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--x", type=float, default=None, help="Sampler x for generation (Saguaro sampling coefficient)")
    
    # Example mode
    parser.add_argument("--example", action="store_true", help="Use real prompts like in example.py and print generations (supports up to batch size 8)")
    parser.add_argument("--humaneval", action="store_true", help="Use HumanEval prompts")
    parser.add_argument("--alpaca", action="store_true", help="Use Alpaca prompts")
    parser.add_argument("--c4", action="store_true", help="Use C4 prompts")
    parser.add_argument("--ultrafeedback", action="store_true", help="Use UltraFeedback prompts")
    parser.add_argument("--random", action="store_true", help="Use random tokens instead of dataset prompts")
    parser.add_argument("--all", action="store_true", help="Use numseqs from each dataset (union dataset with numseqs*4 total)")
    
    # Debugging and logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (saves draft inputs during prefill)")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to wandb")
    parser.add_argument("--group", type=str, default=None, help="Wandb group name")
    parser.add_argument("--name", type=str, default=None, help="Wandb run name")
    
    # Handle eagle implication
    args = parser.parse_args()
    if args.eagle:
        args.spec = True
        assert args.llama, "Eagle currently only supports llama models"
        assert args.temp == 0.0 and args.dtemp is None and args.ttemp is None, "Eagle currently only supports greedy decoding (temp=0)"
        assert getattr(args, 'async', False), "Eagle currently only supports async speculative decoding"
    return args



def create_run_name(args):
    """Create a descriptive run name for wandb logging."""
    spec_mode_str = "spec" if args.spec else "normal"
    async_mode_str = "_async" if getattr(args, 'async', False) else ""
    jit_mode_str = "_jit" if args.backup == "jit" else ""
    model_type = "llama" if args.llama else "qwen"
    example_str = "_example" if args.example else ""
    humaneval_str = "_humaneval" if args.humaneval else ""
    alpaca_str = "_alpaca" if args.alpaca else ""
    c4_str = "_c4" if args.c4 else ""
    ultrafeedback_str = "_ultrafeedback" if args.ultrafeedback else ""
    random_str = "_random" if args.random else ""
    all_str = "_all" if args.all else ""
    gsm_str = "_gsm" if not args.example and not args.humaneval and not args.alpaca and not args.c4 and not args.ultrafeedback and not args.random and not args.all else ""
    sampler_x_str = f"_sampler_x{args.x}" if args.x else ""
    
    # Include all parameters that might be swept
    temp_str = f"_temp{args.temp}"
    if getattr(args, 'async', False):
        if args.dtemp is not None:
            temp_str += f"_dtemp{args.dtemp}"
        if args.ttemp is not None:
            temp_str += f"_ttemp{args.ttemp}"
    
    # Include all parameters from exec_bench.sh that could be swept
    draft_str = f"_draft{args.draft}" if args.draft is not None else "_nodraft"
    k_str = f"_k{args.k}"
    f_str = f"_f{args.f}"
    
    return args.name if args.name else f"{model_type}_size{args.size}_{spec_mode_str}{async_mode_str}{jit_mode_str}_b{args.b}{k_str}{f_str}{draft_str}{temp_str}{sampler_x_str}{example_str}{humaneval_str}{alpaca_str}{c4_str}{ultrafeedback_str}{random_str}{all_str}{gsm_str}"


def initialize_wandb(args, run_name):
    """Initialize wandb logging if requested."""
    if not args.wandb:
        return
    
    wandb.init(
        project="ssd",
        name=run_name,
        group=args.group,
        config={
            "model_size": args.size,
            "gpus": args.gpus,
            "speculative_decoding": args.spec,
            "async_speculative": getattr(args, 'async', False),
            "jit_speculative": args.backup == "jit",
            "k": args.k if args.spec else None,
            "f": args.f,
            "fan_out_list": args.flh,
            "fan_out_list_miss": args.flm,
            "llama": args.llama,
            "max_model_len": args.max_model_len,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "numseqs": args.numseqs,
            "temperature": args.temp,
            "draft_model": args.draft,
            "b": args.b,
            "block_size": args.block_sz,
            "eager": args.eager,
            "example_mode": args.example,
            "humaneval_mode": args.humaneval,
            "alpaca_mode": args.alpaca,
            "c4_mode": args.c4,
            "ultrafeedback_mode": args.ultrafeedback,
            "random_mode": args.random,
            "all_mode": args.all,
            "sampler_x": args.x,
            "implementation": "ssd",
        }
    )


def create_llm_kwargs(args, draft_path):
    """Create LLM initialization arguments."""
    llm_kwargs = dict(
        enforce_eager=args.eager,
        num_gpus=args.gpus,
        speculate=args.spec,
        speculate_k=args.k,
        draft_async=getattr(args, 'async', False),
        async_fan_out=args.f,
        verbose=args.verbose,
        draft=draft_path,
        kvcache_block_size=args.block_sz,
        max_num_seqs=args.b,
        max_model_len=args.max_model_len,
        sampler_x=args.x,
        jit_speculate=(args.backup == "jit"),
    )

    # Pass fan out list if specified
    if args.flh is not None:
        llm_kwargs["fan_out_list"] = args.flh
    if args.flm is not None:
        llm_kwargs["fan_out_list_miss"] = args.flm

    # Pass decoupled async temps when specified
    if getattr(args, 'async', False):
        if args.dtemp is not None:
            llm_kwargs["draft_async_temp"] = args.dtemp
        if args.ttemp is not None:
            llm_kwargs["target_async_temp"] = args.ttemp

    return llm_kwargs


    # removed: helpers moved to bench_helpers


def log_wandb_metrics(args, metrics, total_tokens, total_time, throughput, model_name, mode, run_name):
    """Log metrics to wandb if enabled."""
    if not args.wandb:
        return
    
    wandb_metrics = {
        "official_total_tokens": total_tokens,
        "official_total_time": total_time,
        "official_end_to_end_throughput": throughput,
        "model_name": model_name,
        "mode": mode,
        "run_name": run_name,
    }
    
    # Add internal metrics if available
    if metrics:
        # Calculate and log prefill/decode throughput from metrics
        if "prefill_total_time" in metrics and "prefill_total_tokens" in metrics:
            if metrics["prefill_total_time"] > 0:
                wandb_metrics["metrics_prefill_throughput"] = metrics["prefill_total_tokens"] / metrics["prefill_total_time"]
        
        if "decode_total_time" in metrics and "decode_total_tokens" in metrics:
            if metrics["decode_total_time"] > 0:
                wandb_metrics["metrics_decode_throughput"] = metrics["decode_total_tokens"] / metrics["decode_total_time"]
        
        # Calculate and log average target step time
        if "target_step_times" in metrics and metrics["target_step_times"]:
            avg_target_step_time_ms = sum(metrics["target_step_times"]) * 1000 / len(metrics["target_step_times"])
            wandb_metrics["metrics_avg_target_step_time_ms"] = avg_target_step_time_ms
        
        # Calculate averages for the first three lists
        if "cache_hits" in metrics and metrics["cache_hits"]:
            wandb_metrics["metrics_avg_cache_hits"] = sum(metrics["cache_hits"]) / len(metrics["cache_hits"])
        
        if "accepted_suffix_lens_with_recovery" in metrics and metrics["accepted_suffix_lens_with_recovery"]:
            wandb_metrics["metrics_avg_accepted_suffix_lens_with_recovery"] = sum(metrics["accepted_suffix_lens_with_recovery"]) / len(metrics["accepted_suffix_lens_with_recovery"])
            # Log as histogram
            wandb_metrics["metrics_accepted_suffix_lens_with_recovery_histogram"] = wandb.Histogram(metrics["accepted_suffix_lens_with_recovery"])
        
        if "accepted_suffix_lens_on_hit" in metrics and metrics["accepted_suffix_lens_on_hit"]:
            wandb_metrics["metrics_avg_accepted_suffix_lens_on_hit"] = sum(metrics["accepted_suffix_lens_on_hit"]) / len(metrics["accepted_suffix_lens_on_hit"])
            # Log as histogram
            wandb_metrics["metrics_accepted_suffix_lens_on_hit_histogram"] = wandb.Histogram(metrics["accepted_suffix_lens_on_hit"])

    
    wandb.log(wandb_metrics)


def run_benchmark(args, llm, prompts, sampling_params):
    """Run the actual benchmark and return results."""
    # Log initial progress to wandb
    if args.wandb:
        wandb.log({"sequences_processed": 0, "total_sequences": len(prompts)})

    start_time = time.time()
    outputs, metrics = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time
    
    # Log completion to wandb
    if args.wandb:
        wandb.log({"sequences_processed": len(prompts), "total_sequences": len(prompts)})
    
    return outputs, total_time, metrics


def main():
    args = parse_arguments()
    seed(0)
    
    # Validate example mode constraints
    if args.example and args.numseqs > 8:
        print("Warning: --example mode supports up to 8 sequences, reducing numseqs from {} to 8".format(args.numseqs))
        args.numseqs = 8
    
    # Get model paths
    model_name, model_path, draft_path = get_model_paths(args)
    
    # Prepare inputs (string prompts or token ids) before setting up LLM
    string_prompts, prompt_token_ids, original_prompts = generate_benchmark_inputs(args, model_path)
    prompts = string_prompts if string_prompts is not None else prompt_token_ids
    
    # Build sampling params matching internal engine (vanilla sampling)
    if prompts:
        num_reqs = len(prompts)
    else:
        num_reqs = args.numseqs
    sampling_params = [SamplingParams(
        temperature=args.temp,
        ignore_eos=True,
        max_new_tokens=args.output_len,
        draft_async_temperature=(args.dtemp if getattr(args, 'async', False) and args.dtemp is not None else None),
        target_async_temperature=(args.ttemp if getattr(args, 'async', False) and args.ttemp is not None else None),
    ) for _ in range(num_reqs)]
    
    # Print number of tokens in each input
    if prompts:
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                # Load tokenizer to count tokens
                print(f'Prompt: {prompt}')
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                num_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            elif isinstance(prompt, list):
                # Already tokenized
                num_tokens = len(prompt)
            else:
                raise ValueError(f"Invalid prompt type: {type(prompt)}")
    
    # Create run name and initialize wandb
    run_name = create_run_name(args)
    initialize_wandb(args, run_name)
    
    # Create LLM
    llm_kwargs = create_llm_kwargs(args, draft_path)
    if args.eagle:
        llm_kwargs['use_eagle'] = True
    if args.debug:
        llm_kwargs['debug_mode'] = True
        
    llm = LLM(model_path, **llm_kwargs)
    
    try:
        # Run benchmark
        outputs, total_time, metrics = run_benchmark(args, llm, prompts, sampling_params)
        
        # Calculate and display results
        total_tokens = sum(sp.max_new_tokens for sp in sampling_params)
        throughput = total_tokens / total_time
        
        mode = "Eager" if args.eager else "CUDA Graphs"
        spec_mode = f" + Speculative(k={args.k})" if args.spec else ""
        async_mode = " + Async" if getattr(args, 'async', False) else ""
        jit_mode = " + JIT" if args.backup == "jit" else ""
        x_mode = f" + X({args.x})" if args.x else ""
        full_mode = mode + spec_mode + async_mode + jit_mode + x_mode
        
        print(f"Model: {model_name}, Mode: {full_mode}, Total: {total_tokens}tok, Time: {total_time:.2f}s, Total Throughput: {throughput:.2f}tok/s")
        
        # Print generations unless in random mode
        if not args.random:
            print("\n" + "="*80)
            print("GENERATIONS:")
            print("="*80)
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Print at most 10 generations
            for i, (prompt, output) in enumerate(zip(prompts, outputs)):
                if i >= 10:
                    break
                    
                # Decode prompt if it's tokenized
                if isinstance(prompt, list):
                    decoded_prompt = tokenizer.decode(prompt, skip_special_tokens=True)
                else:
                    decoded_prompt = prompt
                
                # Use original prompt if available, otherwise decoded prompt
                if original_prompts and i < len(original_prompts):
                    display_prompt = original_prompts[i]
                else:
                    display_prompt = decoded_prompt
                
                print(f"\nPrompt {i+1}: {display_prompt!r}")
                print(f"Generation: {output['text']!r}")
                print("-" * 40)
        
        # Log metrics to wandb
        log_wandb_metrics(args, metrics, total_tokens, total_time, throughput, model_name, full_mode, run_name)
        
        # Finish wandb run
        if args.wandb:
            wandb.finish()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    
    print(f'Engine exited!')

    sys.exit(0)


if __name__ == "__main__":
    main()

