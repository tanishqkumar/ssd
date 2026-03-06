"""Interactive chat with SSD, SGLang, or vLLM.

Usage:
    python chat.py --ssd --spec --async --k 7 --f 2 --gpus 5
    python chat.py --ssd --spec --k 7 --gpus 4
    python chat.py --sglang [--ar]
    python chat.py --vllm [--ar]
    python chat.py --sglang --url http://localhost:40010
"""

import os, sys, time, json, signal, argparse, subprocess, multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from bench_paths import HF_CACHE_DIR, resolve_snapshot
from server_lifecycle import ensure_port_available

TARGET = resolve_snapshot(f"{HF_CACHE_DIR}/models--meta-llama--Llama-3.1-70B-Instruct")
DRAFT = resolve_snapshot(f"{HF_CACHE_DIR}/models--meta-llama--Llama-3.2-1B-Instruct")


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ssd", action="store_true")
    g.add_argument("--sglang", action="store_true")
    g.add_argument("--vllm", action="store_true")
    p.add_argument("--temp", type=float, default=0)
    p.add_argument("--output_len", type=int, default=2048)
    p.add_argument("--ar", action="store_true")
    # SSD
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--spec", action="store_true")
    p.add_argument("--async", action="store_true", dest="async_spec")
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--f", type=int, default=2)
    p.add_argument("--backup", type=str, default="jit")
    p.add_argument("--b", type=int, default=1)
    p.add_argument("--x", type=float, default=None)
    p.add_argument("--eager", action="store_true")
    p.add_argument(
        "--ignore_eos",
        action="store_true",
        help="Ignore EOS token (generate full output_len)",
    )
    p.add_argument(
        "--metrics",
        action="store_true",
        help="Print token count, speed, TTFT after each response",
    )
    # Server
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--url", type=str, default=None)
    return p.parse_args()


def _decode_worker(conn, tokenizer):
    """Runs in subprocess to avoid GIL contention with CUDA."""
    ids, prev = [], ""
    while True:
        chunk = conn.recv()
        if chunk is None:
            break
        ids.extend(chunk)
        full = tokenizer.decode(ids, skip_special_tokens=True)
        delta = full[len(prev) :]
        if delta:
            print(delta, end="", flush=True)
        prev = full


def ssd_chat(args):
    import ssd.paths  # noqa
    from ssd import LLM, SamplingParams
    from transformers import AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    llm = LLM(
        TARGET,
        enforce_eager=args.eager,
        num_gpus=args.gpus,
        speculate=args.spec,
        speculate_k=args.k,
        draft_async=args.async_spec,
        async_fan_out=args.f,
        draft=DRAFT,
        kvcache_block_size=256,
        max_num_seqs=args.b,
        max_model_len=8192,
        sampler_x=args.x,
        jit_speculate=(args.backup == "jit"),
    )

    history = []
    print(f"\nChat via SSD. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("User: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        history.append({"role": "user", "content": user_input})
        token_ids = tokenizer.apply_chat_template(
            history, add_generation_prompt=True, tokenize=True
        )
        sp = SamplingParams(
            temperature=args.temp,
            max_new_tokens=args.output_len,
            ignore_eos=args.ignore_eos,
        )

        parent_conn, child_conn = mp.Pipe()
        proc = mp.Process(
            target=_decode_worker, args=(child_conn, tokenizer), daemon=True
        )
        proc.start()
        child_conn.close()

        t_first = [None]

        def on_tokens(seq_id, new_ids):
            if t_first[0] is None:
                t_first[0] = time.time()
            parent_conn.send(list(new_ids))

        print("Assistant: ", end="", flush=True)
        t0 = time.time()
        outputs, _ = llm.generate(
            [token_ids], [sp], use_tqdm=False, stream_callback=on_tokens
        )
        t_end = time.time()
        parent_conn.send(None)
        proc.join(timeout=5)

        n = len(outputs[0]["token_ids"])
        ttft = (t_first[0] - t0) if t_first[0] else 0
        dt = t_end - (t_first[0] or t0)
        if args.metrics:
            print(f"\n  [{n} tok, {n / dt:.0f} tok/s, {ttft:.2f}s TTFT]")
        print()
        history.append({"role": "assistant", "content": outputs[0]["text"]})


def wait_for_server(port, timeout=900, interval=5):
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if (
                requests.get(f"http://localhost:{port}/health", timeout=2).status_code
                == 200
            ):
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(interval)
    return False


def server_chat(args):
    import requests
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET)

    backend = "sglang" if args.sglang else "vllm"
    port = args.port or (40010 if args.sglang else 40020)
    proc = None

    if args.url:
        base_url = args.url.rstrip("/")
    else:
        if args.sglang:
            cmd = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                TARGET,
                "--tp",
                str(args.tp),
                "--mem-fraction-static",
                "0.70",
                "--max-running-requests",
                "1",
                "--disable-radix-cache",
                "--log-level",
                "warning",
                "--port",
                str(port),
            ]
            if not args.ar:
                cmd += [
                    "--speculative-algorithm",
                    "STANDALONE",
                    "--speculative-draft-model-path",
                    DRAFT,
                    "--speculative-num-steps",
                    "4",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "5",
                ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                TARGET,
                "--tensor-parallel-size",
                str(args.tp),
                "--gpu-memory-utilization",
                "0.90",
                "--max-num-seqs",
                "1",
                "--disable-log-requests",
                "--disable-log-stats",
                "--port",
                str(port),
            ]
            if not args.ar:
                spec = {
                    "model": DRAFT,
                    "num_speculative_tokens": 5,
                    "method": "draft_model",
                }
                cmd += ["--speculative-config", json.dumps(spec)]

        ensure_port_available(port, backend)
        print(f"Launching {backend}...")
        proc = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not wait_for_server(port):
            print("Server failed to start")
            sys.exit(1)
        print("Ready.")
        base_url = f"http://localhost:{port}"

    print(f"\nChat via {backend.upper()}. Type 'quit' to exit.\n")
    history = []

    try:
        while True:
            try:
                user_input = input("User: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_input.strip().lower() in ("quit", "exit", "q"):
                break

            history.append({"role": "user", "content": user_input})
            token_ids = tokenizer.apply_chat_template(
                history, add_generation_prompt=True, tokenize=True
            )
            payload = {
                "model": TARGET,
                "prompt": token_ids,
                "temperature": args.temp,
                "max_tokens": args.output_len,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

            print("Assistant: ", end="", flush=True)
            t0 = time.time()
            resp = requests.post(
                f"{base_url}/v1/completions", json=payload, stream=True
            )
            resp.raise_for_status()

            full, t_first, usage = "", None, {}
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = (
                    chunk["choices"][0].get("text", "") if chunk.get("choices") else ""
                )
                if delta:
                    if t_first is None:
                        t_first = time.time()
                    full += delta
                    print(delta, end="", flush=True)
                if "usage" in chunk:
                    usage = chunk["usage"]

            t_end = time.time()
            n = usage.get("completion_tokens", len(full.split()))
            ttft = (t_first - t0) if t_first else 0
            dt = t_end - (t_first or t0)
            if args.metrics:
                print(f"\n  [{n} tok, {n / dt:.0f} tok/s, {ttft:.2f}s TTFT]")
            print()
            history.append({"role": "assistant", "content": full})
    finally:
        if proc:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()


if __name__ == "__main__":
    args = parse_args()
    ssd_chat(args) if args.ssd else server_chat(args)
