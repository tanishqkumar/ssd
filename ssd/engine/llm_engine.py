import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'  # for FlashInfer

from ssd.config import Config
from ssd.sampling_params import SamplingParams
from ssd.utils.misc import infer_model_family
from ssd.engine.sequence import Sequence
from ssd.engine.scheduler import Scheduler
from ssd.engine.model_runner import ModelRunner
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.speculator_async import SpeculatorAsync
from ssd.engine.speculator_sync import SpeculatorSync
from ssd.engine.step import InferenceStep, AutoRegressiveStep, SpecDecodeStep
from ssd.engine.verifier import Verifier

import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp



METRICS = {
    "cache_hits": [],
    "accepted_suffix_lens_with_recovery": [],
    "accepted_suffix_lens_on_hit": [],  # Only for cache hits in async mode
    "accepted_suffix_lens_on_miss": [],  # Only for cache misses in async mode
    "prefill_total_time": 0,
    "decode_total_time": 0,
    "prefill_total_tokens": 0,
    "decode_total_tokens": 0,
    "target_step_times": [],
    "num_spec_steps": [],
}


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size 

        assert config.kvcache_block_size >= (
            2 * config.speculate_k + 2), "ERROR: support for block size < 2*k+2 is not implemented"
        assert config.num_gpus > 1 or not config.draft_async, "ERROR: draft_async requires at least 2 gpus"
            
        # Check that target and draft are from the same family
        if config.speculate:
            target_family = infer_model_family(config.model)
            draft_family = infer_model_family(config.draft)
            assert target_family == draft_family, f"ERROR: target model family and draft model family must match"

        self.ps = []
        self.events = []

        ctx = mp.get_context("spawn")
        self.num_tp_gpus = config.num_gpus if not self.config.draft_async else config.num_gpus - 1

        if config.speculate and config.draft_async:
            self.draft_ps = None

        for i in range(1, self.num_tp_gpus):
            if self.config.verbose:
                print(f'creating ModelRunner process {i}', flush=True)
            event = ctx.Event()
            # can't pass kwargs through ctx.Process args
            process = ctx.Process(target=ModelRunner, args=(
                config, i, event, False, self.num_tp_gpus))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        if self.config.verbose:
            print(
                f'config.speculate = {config.speculate} and config.draft_async = {config.draft_async} about to create draft runner', flush=True)

        if config.speculate and config.draft_async:
            init_q = ctx.Queue()
            draft_rank = config.num_gpus - 1
            self.draft_ps = ctx.Process(
                target=DraftRunner, args=(config, draft_rank, init_q))
            self.draft_ps.start()
            print(
                f'Draft runner created on rank {draft_rank} (async)!', flush=True)

        # modelRunner(0) will wait on all 5 processes, so other 4 need to have launched by now
        self.model_runner = ModelRunner(
            config, 0, self.events, is_draft=False, num_tp_gpus=self.num_tp_gpus)

        # do this after so we can launch model runner above so that the q is actually populated
        if config.speculate and config.draft_async:
            try:
                num_blocks = init_q.get(timeout=180)  # seconds
            except Exception as e:
                raise RuntimeError(
                    "ERROR: Timed out waiting for draft kv cache size") from e

            init_q.close()
            self.draft_cfg = DraftRunner.create_draft_config(config)
            self.draft_cfg.num_kvcache_blocks = num_blocks  # set for block manager to knwo

            self.prev_allocated_blocks = None
            self.prev_blocks_per_fork = None

        if config.speculate and not config.draft_async:
            # keep it colocated on rank 0, process/dist agnostic in this case
            self.draft_runner = DraftRunner(config)
            self.draft_cfg = self.draft_runner.draft_cfg
            print(f'Draft runner created on rank 0 (no async)', flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config, draft_cfg=self.draft_cfg if config.speculate else None)
        assert config.max_model_len == self.scheduler.max_model_len

        print(f"[LLMEngine] finished llm_engine init", flush=True)

        self._exiting = False
        atexit.register(lambda: self.exit(hard=True))

    def exit(self, hard: bool = True):
        print(f"[LLMEngine] Exiting (hard={hard})", flush=True)
        if getattr(self, "_exiting", False):
            return
        self._exiting = True
        # 1) If async, tell draft to quit before tearing down anything
        try:
            if self.config.speculate and self.config.draft_async:
                # Use local method (no SHM) to send cmd=2
                self.model_runner.send_draft_exit_signal()
        except Exception:
            pass
        # 2) Tell all target ranks (including rank 0 self) to exit (non-blocking cleanup, no os._exit inside)
        try:
            self.model_runner.call("exit",
                                   True if not self.config.draft_async else True)
        except Exception:
            pass
        # 3) Wait briefly for TP workers; terminate if still around
        try:
            if self.model_runner.world_size > 1:
                for p in self.ps:
                    p.join(timeout=3)
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=2)
        except Exception:
            pass
        # 4) Draft process: after sending cmd=2, give it a moment, then terminate if needed
        try:
            if self.config.speculate and self.config.draft_async and self.draft_ps is not None:
                self.draft_ps.join(timeout=3)
                if self.draft_ps.is_alive():
                    self.draft_ps.terminate()
                    self.draft_ps.join(timeout=2)
        except Exception:
            pass
        # 5) Force-exit current process if requested
        if hard:
            os._exit(0)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)


    def step(self, step: InferenceStep):
        t = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        ttl_tokens = step.prefill(seqs) if is_prefill else step.decode(seqs)

        time_taken = perf_counter() - t

        if is_prefill:
            METRICS["prefill_total_time"] += time_taken
            METRICS["prefill_total_tokens"] += ttl_tokens
        else:
            METRICS["decode_total_time"] += time_taken
            METRICS["decode_total_tokens"] += ttl_tokens

        outputs = [(seq.seq_id, seq.completion_token_ids)
                   for seq in seqs if seq.is_finished]

        return outputs

    def is_finished(self):
        return self.scheduler.is_finished()

    def log_metrics(self):
        avg_prefill_throughput = METRICS["prefill_total_tokens"] / \
            METRICS["prefill_total_time"]
        avg_decode_throughput = METRICS["decode_total_tokens"] / \
            METRICS["decode_total_time"]
        print(
            f"Final Prefill Throughput: {int(avg_prefill_throughput)}tok/s", flush=True)
        print(
            f"Final Decode Throughput: {int(avg_decode_throughput)}tok/s", flush=True)

        if self.config.speculate:
            ttl_accepted_with_recovery = sum(METRICS['accepted_suffix_lens_with_recovery'])
            ttl_num_spec_steps = len(METRICS['accepted_suffix_lens_with_recovery'])
            avg_tokens_per_step = ttl_accepted_with_recovery / ttl_num_spec_steps
            print(
                f"[metrics] Avg Tokens per step (incl recovery): {avg_tokens_per_step:.2f}", flush=True)

            total_accepted = ttl_accepted_with_recovery - ttl_num_spec_steps
            avg_acceptance_rate = (total_accepted / ttl_num_spec_steps) / self.config.speculate_k
            print(
                f"[metrics] Avg Fraction of Speculated Tokens Accepted: {avg_acceptance_rate:.2f}", flush=True)
            print(
                f"[metrics] Avg target time per full step (ms): {sum(METRICS['target_step_times']) * 1000 / len(METRICS['target_step_times']):.2f}", flush=True)
            if self.config.draft_async:
                print(
                    f"[metrics] Avg Cache Hits: {sum(METRICS['cache_hits']) / len(METRICS['cache_hits']):.2f}", flush=True)
                # Log separate metrics for cache hits
                if METRICS['accepted_suffix_lens_on_hit']:
                    avg_suffix_len_on_hit = sum(
                        METRICS['accepted_suffix_lens_on_hit']) / len(METRICS['accepted_suffix_lens_on_hit'])
                    print(
                        f"[metrics] Avg Tokens per step on Cache Hit: {avg_suffix_len_on_hit:.2f}", flush=True)
                    
                    # Calculate empirical frequencies of accepted_suffix_lens_on_hit - 1
                    adjusted_lens = [length - 1 for length in METRICS['accepted_suffix_lens_on_hit']]
                    total_count = len(adjusted_lens)
                    freq_counts = {}
                    for length in adjusted_lens:
                        freq_counts[length] = freq_counts.get(length, 0) + 1
                    
                    # Print normalized empirical probabilities for range [0, K]
                    print(f"[metrics] Empirical frequencies of accepted_suffix_lens_on_hit - 1:", flush=True)
                    for k in range(self.config.speculate_k + 1):
                        prob = freq_counts.get(k, 0) / total_count
                        print(f"  {k}: {prob:.3f}", flush=True)
                if METRICS['accepted_suffix_lens_on_miss']:
                    avg_suffix_len_on_miss = sum(
                        METRICS['accepted_suffix_lens_on_miss']) / len(METRICS['accepted_suffix_lens_on_miss'])
                    print(
                        f"[metrics] Avg Tokens per step on Cache Miss: {avg_suffix_len_on_miss:.2f}", flush=True)
                else:
                    print(
                        f"[metrics] Avg Tokens per step on Cache Hit: N/A (no cache hits)", flush=True)

    def create_inference_step(self, config: Config) -> InferenceStep:
        if config.speculate:
            if config.draft_async:
                speculator = SpeculatorAsync(
                    lookahead=config.speculate_k,
                    device=config.device,
                    async_fan_out=config.async_fan_out,
                    max_blocks=config.max_blocks,
                    vocab_size=config.draft_hf_config.vocab_size,
                    draft_dtype=config.draft_hf_config.torch_dtype,
                    kvcache_block_size=config.kvcache_block_size,
                    max_model_len=config.max_model_len,
                    async_pg=self.model_runner.async_pg,
                    draft_runner_rank=self.num_tp_gpus,
                )
            else:
                speculator = SpeculatorSync(
                    lookahead=config.speculate_k,
                    device=config.device,
                    draft_model_runner=self.draft_runner,
                )

            verifier = Verifier(
                lookahead=config.speculate_k,
                device=config.device,
                target_model_runner=self.model_runner,
                sampler_x=config.sampler_x,
                async_fan_out=config.async_fan_out,
                jit_speculate=config.jit_speculate,
                tokenizer=self.tokenizer,
                metrics=METRICS,
            )
            return SpecDecodeStep(
                scheduler=self.scheduler,
                speculator=speculator,
                verifier=verifier,
                eagle=config.use_eagle,
                tokenizer=self.tokenizer,
            )
        else:
            return AutoRegressiveStep(
                scheduler=self.scheduler,
                model_runner=self.model_runner,
                tokenizer=self.tokenizer,
            )

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        
        if use_tqdm:
            pbar = tqdm(total=len(prompts),
                        desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        inference_step = self.create_inference_step(self.config)
        i = 0
        max_steps = self.config.max_steps if self.config.max_steps is not None else float('inf')
        while not self.is_finished() and i < max_steps:
            if self.config.verbose:
                print(f"[generate] step {i+1}" +
                    f" of {max_steps}" if max_steps != float('inf') else "",
                    flush=True,
                )
            t = perf_counter()
            output = self.step(inference_step)
            time_taken = perf_counter() - t
            METRICS["target_step_times"].append(time_taken)

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(
            token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()

        self.log_metrics()

        return outputs, METRICS

