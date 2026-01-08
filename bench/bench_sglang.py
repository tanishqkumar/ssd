import os
import time
import argparse
from random import seed
from typing import Tuple
import sys
import json
from transformers import AutoTokenizer
import wandb
from bench_helpers import get_model_paths, generate_benchmark_inputs, DATASET_PATHS

# Set CUDA architecture for H100
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang with Qwen/Llama models and optional speculative decoding.")

    # Model and infrastructure
    parser.add_argument("--size", type=str, choices=["0.6", "1.7", "4", "8", "14", "32", "1", "3", "70"], default="0.6",
                        help="Model size in billions of parameters")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated GPU indices to use (e.g., '0,1,2,3'). Overrides auto selection.")
    parser.add_argument("--auto-select-gpus", action="store_true",
                        help="Automatically select top-N GPUs by free memory for TP (uses NVML if available)")
    parser.add_argument("--llama", action="store_true",
                        help="Use Llama models instead of Qwen")

    # Speculative decoding
    parser.add_argument("--spec", action="store_true",
                        help="Enable speculative decoding")
    parser.add_argument("--spec-algo", type=str, default="STANDALONE", choices=["STANDALONE", "EAGLE3"],
                        help="Speculative decoding algorithm")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of speculative tokens (lookahead)")
    parser.add_argument("--draft", type=str, default=None,
                        help="Draft model size (e.g., 1 for Llama-1B, 0.6 for Qwen-0.6B) or full path to draft model")

    # Generation parameters
    parser.add_argument("--input_len", type=int, default=128,
                        help="Prompt length in tokens")
    parser.add_argument("--output_len", type=int, default=256, 
                        help="New tokens to generate per request")
    parser.add_argument("--numseqs", type=int, default=1,
                        help="Number of requests to generate (batch size)")
    parser.add_argument("--temp", type=float, default=0.0,
                        help="Temperature for generation")

    # Other parameters
    parser.add_argument("--b", type=int, default=1,
                        help="Max in-flight sequences")
    parser.add_argument("--gpu-mem-util", type=float,
                        default=0.9, help="GPU memory utilization (0-1)")

    # Dataset selection
    parser.add_argument("--example", action="store_true",
                        help="Use real prompts like in example.py and print generations")
    parser.add_argument("--humaneval", action="store_true",
                        help="Use HumanEval prompts")
    parser.add_argument("--alpaca", action="store_true",
                        help="Use Alpaca prompts")
    parser.add_argument("--c4", action="store_true", help="Use C4 prompts")
    parser.add_argument("--ultrafeedback", action="store_true",
                        help="Use UltraFeedback prompts")
    parser.add_argument("--random", action="store_true",
                        help="Use random tokens instead of dataset prompts")
    parser.add_argument("--all", action="store_true",
                        help="Use numseqs from each dataset (union dataset with numseqs*4 total)")

    # Debugging and logging
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--wandb", action="store_true",
                        help="Log metrics to wandb")
    parser.add_argument("--group", type=str, default=None,
                        help="Wandb group name")
    parser.add_argument("--name", type=str, default=None,
                        help="Wandb run name")

    args = parser.parse_args()

    # Get model paths
    model_name, model_path, draft_path = get_model_paths(args)

    print(f"Using model: {model_path}")
    if args.spec and draft_path:
        print(f"Using draft model: {draft_path}")

    # Set CUDA memory allocation configuration to avoid fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Optionally select GPUs before importing SGLang so distributed init sees the right devices
    def _set_cuda_visible_devices_from_args(args) -> None:
        if args.gpus <= 0:
            return
        # Explicit device list takes precedence
        if args.devices:
            devs = ",".join([d.strip()
                            for d in args.devices.split(",") if d.strip() != ""])
            if devs:
                os.environ["CUDA_VISIBLE_DEVICES"] = devs
                print(f"Using GPUs (explicit): {devs}")
            return
        # Auto-select by free memory when multi-GPU and no explicit devices
        if args.gpus > 1 and (args.auto_select_gpus or not args.devices):
            try:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                mem_list = []
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_list.append((i, mem.free, mem.total))
                # Sort by free memory desc
                mem_list.sort(key=lambda x: x[1], reverse=True)
                selected = [str(idx) for idx, _, _ in mem_list[: args.gpus]]
                if len(selected) >= args.gpus:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
                    print(
                        f"Using GPUs (auto-selected by free memory): {os.environ['CUDA_VISIBLE_DEVICES']}")
                pynvml.nvmlShutdown()
            except Exception as e:
                print(
                    f"Warning: GPU auto-selection failed: {e}. Proceeding with default device visibility.")

    _set_cuda_visible_devices_from_args(args)

    # Import SGLang after setting CUDA_VISIBLE_DEVICES
    import sglang as sgl

    # Initialize SGLang offline engine
    engine_args = {
        "model_path": model_path,
        "tp_size": args.gpus,
        "mem_fraction_static": args.gpu_mem_util,
        "cuda_graph_max_bs": 1,
        "disable_cuda_graph": True,
        "max_running_requests": args.b,
    }

    # Add speculative decoding if enabled
    if args.spec and draft_path:
        engine_args["speculative_algorithm"] = args.spec_algo
        engine_args["speculative_eagle_topk"] = None  # ignored
        
        # For EAGLE3, use specific draft models based on target size
        if args.spec_algo == "EAGLE3":
            # engine_args["disable_custom_all_reduce"] = True
            engine_args["dtype"] = "bfloat16"
            os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
            if not args.llama:
                raise ValueError("EAGLE3 requires --llama flag to be set")
            
            # Determine target model size from model_path
            if "70B" in model_path or "70b" in model_path:
                draft_path = "lmsys/sglang-EAGLE3-LLaMA3.3-Instruct-70B"
            elif "8B" in model_path or "8b" in model_path:
                draft_path = "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
            else:
                raise ValueError(f"EAGLE3 only supports 8B and 70B Llama models, but got model_path: {model_path}")
        
        engine_args["speculative_draft_model_path"] = draft_path
        # engine_args["speculative_num_draft_tokens"] = args.k

    print(args.spec, engine_args, flush=True)
    llm = sgl.Engine(**engine_args)

    # Generate benchmark inputs
    string_prompts, prompt_token_ids, original_prompts = generate_benchmark_inputs(
        args, model_path)

    # Use string prompts for SGLang (it expects strings, not token IDs)
    if string_prompts is not None:
        prompts = string_prompts
    elif prompt_token_ids is not None:
        # Convert token IDs back to strings using tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        prompts = [tokenizer.decode(token_ids, skip_special_tokens=False)
                    for token_ids in prompt_token_ids]
    else:
        raise ValueError("No prompts generated")

    print(f"Generated {args.b} prompts")
    if args.verbose and args.b > 0:
        print(f"Sample prompt: {prompts[0][:100]}...")

    # Set up sampling parameters
    sampling_params = {
        "temperature": args.temp,
        "max_new_tokens": args.output_len,
    }

    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project="sglang-benchmark",
            group=args.group,
            name=args.name,
            config=vars(args)
        )

    try:
        # Warmup run
        warmup_prompt = prompts[0]
        _ = llm.generate([warmup_prompt], sampling_params)
        print("Warmup completed, about to run actual benchmark")

        # Benchmark generation
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()

        total_time = end_time - start_time

        # Calculate metrics
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        total_prompt_tokens = 0
        total_completion_tokens = 0

        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            prompt_tokens = len(tokenizer.encode(prompt))
            # SGLang outputs can be strings or objects with 'text' attribute
            if isinstance(output, str):
                completion_text = output
            elif hasattr(output, 'text'):
                completion_text = output.text
            elif isinstance(output, dict) and 'text' in output:
                completion_text = output['text']
            else:
                completion_text = str(output)

            completion_tokens = len(tokenizer.encode(
                completion_text)) - prompt_tokens

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            if args.example and i < 3:  # Show first 3 examples
                print(f"\n--- Example {i+1} ---")
                print(f"Prompt: {prompt[:100]}...")
                print(f"Generated: {completion_text[:200]}...")

        # Calculate throughput metrics
        total_tokens = total_prompt_tokens + total_completion_tokens
        throughput_total = total_tokens / total_time
        throughput_output = total_completion_tokens / total_time

        # Print results
        print(f"\n=== Benchmark Results ===")
        print(f"Model: {model_name}")
        print(f"Batch size: {args.b}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total completion tokens: {total_completion_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Throughput (total): {throughput_total:.2f} tokens/sec")
        print(f"Throughput (output): {throughput_output:.2f} tokens/sec")
        print(f"Latency per request: {total_time / len(prompts):.3f} seconds")

        if args.spec:
            print(
                f"Speculative decoding enabled (algo={args.spec_algo}, k={args.k})")

        # Log to wandb if enabled
        if args.wandb:
            wandb.log({
                "total_time": total_time,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "throughput_total": throughput_total,
                "throughput_output": throughput_output,
                "latency_per_request": total_time / len(prompts),
                "batch_size": args.b,
                "speculative_enabled": args.spec,
                "speculative_algo": args.spec_algo if args.spec else None,
                "speculative_k": args.k if args.spec else 0,
            })

    except Exception as e:
        print(f"Error during generation: {e}")
        if args.wandb:
            wandb.log({"error": str(e)})
    finally:
        # Shutdown the engine
        if 'llm' in locals():
            llm.shutdown()
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
