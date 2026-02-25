import time
import argparse
import sys
import asyncio
import aiohttp
import requests
from random import seed
from transformers import AutoTokenizer
from bench_helpers import get_model_paths, generate_benchmark_inputs

async def send_request(session, url, prompt, sampling_params):
    payload = {"text": prompt, "sampling_params": sampling_params}
    async with session.post(f"{url}/generate", json=payload) as resp:
        assert resp.status == 200, f"Request failed: {resp.status}"
        return await resp.json()

async def run_batch_async(url, prompts, sampling_params):
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, url, p, sampling_params) for p in prompts]
        start = time.time()
        results = await asyncio.gather(*tasks)
        return results, time.time() - start

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=40010)
    parser.add_argument("--size", type=str, default="70")
    parser.add_argument("--llama", action="store_true", default=True)
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--draft", type=str, default="1")
    parser.add_argument("--input_len", type=int, default=128)
    parser.add_argument("--output_len", type=int, default=512)
    parser.add_argument("--numseqs", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--humaneval", action="store_true")
    parser.add_argument("--alpaca", action="store_true")
    parser.add_argument("--c4", action="store_true")
    parser.add_argument("--ultrafeedback", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--eagle", action="store_true")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.qwen:
        args.llama = False
    seed(0)

    url = f"http://{args.host}:{args.port}"
    r = requests.get(f"{url}/health", timeout=10)
    assert r.status_code == 200, f"Server not reachable at {url}"
    print(f"Server OK at {url}")

    _, model_path, _ = get_model_paths(args)
    string_prompts, prompt_token_ids, _ = generate_benchmark_inputs(args, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if string_prompts is not None:
        text_prompts = string_prompts
    else:
        text_prompts = [tokenizer.decode(tids[:args.input_len], skip_special_tokens=True)
                        for tids in prompt_token_ids]

    print(f"Prepared {len(text_prompts)} prompts")

    sampling_params = {"temperature": args.temp, "max_new_tokens": args.output_len, "ignore_eos": True}

    if args.wandb:
        import wandb
        wandb.init(project="ssd", group=args.group, name=args.name, config=vars(args))

    requests.post(f"{url}/generate", json={"text": text_prompts[0], "sampling_params": sampling_params}, timeout=120)
    print("Warmup done")

    all_results = []
    total_time = 0
    num_batches = (len(text_prompts) + args.b - 1) // args.b

    for i in range(0, len(text_prompts), args.b):
        batch = text_prompts[i:i + args.b]
        batch_num = i // args.b + 1
        print(f"Batch {batch_num}/{num_batches} ({len(batch)} reqs)...", end=" ", flush=True)
        results, elapsed = asyncio.run(run_batch_async(url, batch, sampling_params))
        all_results.extend(zip(batch, results))
        total_time += elapsed
        print(f"{elapsed:.1f}s")

    total_prompt_tokens = 0
    total_completion_tokens = 0
    for prompt, output in all_results:
        total_prompt_tokens += len(tokenizer.encode(prompt, add_special_tokens=False))
        text = output.get("text", "")
        if text.startswith(prompt):
            text = text[len(prompt):]
        total_completion_tokens += len(tokenizer.encode(text, add_special_tokens=False))

    total_tokens = total_prompt_tokens + total_completion_tokens
    decode_tps = total_completion_tokens / total_time
    total_tps = total_tokens / total_time

    print(f"\n=== Results ===")
    print(f"Requests: {len(all_results)}, Batch size: {args.b}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Prompt tokens: {total_prompt_tokens}, Completion tokens: {total_completion_tokens}")
    print(f"Decode throughput: {decode_tps:.1f} tok/s")
    print(f"Total throughput: {total_tps:.1f} tok/s")
    print(f"Avg latency/req: {total_time / len(all_results):.3f}s")

    if args.wandb:
        import wandb
        wandb.log({
            "metrics_decode_throughput": decode_tps,
            "official_end_to_end_throughput": total_tps,
            "total_completion_tokens": total_completion_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_time": total_time,
        })
        wandb.finish()

if __name__ == "__main__":
    main()
