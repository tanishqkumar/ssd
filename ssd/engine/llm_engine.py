import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'  # for FlashInfer

from ssd.engine.helpers.handshake_helpers import TargetDraftHandshake
from ssd.config import Config
from ssd.sampling_params import SamplingParams
from ssd.utils.misc import infer_model_family
from ssd.engine.sequence import Sequence
from ssd.engine.scheduler import Scheduler
from ssd.engine.model_runner import ModelRunner
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.helpers.runner_helpers import prepare_prefill_tensors_from_seqs
from ssd.utils.verify import verify

import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp
import torch.distributed as dist



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

    # make and send [cache_keys, num_tokens, dbt, temps] tensors, since no Seq objects can be sent via NCCL
    # recv [cache_hits, spec_tokens_flat, logits_q], contract specified in handshake_helpers.py
    def target_draft_handshake(self, seqs_copy: list[Sequence]):
        """Send a spec request command with cache keys to get speculations/logits."""
        draft_runner_rank = self.num_tp_gpus

        # Use the handshake helper to handle the complete protocol
        handshake = TargetDraftHandshake(
            seqs_copy,
            self.config,
            self.model_runner.async_pg,
            draft_runner_rank
        )

        cache_hits, speculations, logits_q = handshake.execute_full_handshake(
            self.draft_cfg)

        METRICS["cache_hits"].append(
            cache_hits.to(torch.float32).mean().item())
        return cache_hits, speculations, logits_q

    def speculate_async(self, seqs: list[Sequence]):
        """
        - Hit the cache with cache_keys, get speculations and logits
            - This will involve deallocating past failed fork blocks, and allocating new forks, and coordinating this with the draft runner
                - But this is abstracted away at this level 
        - Send info to help DraftRunner build tree cache (everything make_branch_bt needs)
            - This includes using helpers that eg. allocate blocks for the forked sequences 
            - And sending context we built to DraftRunner via nccl 
        """
        seqs_copy = [seq.clone_spec() for seq in seqs]

        # Append recovery tokens to local seqs_copy to correctly calculate initial positions
        for seq in seqs_copy:
            if seq.recovery_token_id is None:
                raise ValueError(
                    f"recovery_token_id is None for seq {seq.seq_id}")
            seq.append_token(seq.recovery_token_id)
        
        # Log sequence trunk state for debugging
        if self.config.verbose:
            print(f"\n{'='*80}", flush=True)
            print(f"[TARGET SEQUENCE TRUNK] Batch size: {len(seqs)}", flush=True)
            for i, seq in enumerate(seqs):
                # Show last 20 tokens of trunk + recovery token
                trunk_tokens = seq.token_ids[-20:] if len(seq.token_ids) > 20 else seq.token_ids
                trunk_text = self.tokenizer.decode(trunk_tokens, skip_special_tokens=False)
                recovery_text = self.tokenizer.decode([seq.recovery_token_id], skip_special_tokens=False)
                print(f"  Seq {seq.seq_id} (len={len(seq.token_ids)}):", flush=True)
                print(f"    Trunk (last 20): ...{trunk_text}", flush=True)
                print(f"    Recovery token: {seq.recovery_token_id} ('{recovery_text}')", flush=True)
            print(f"{'='*80}\n", flush=True)

        # hit draft tree cache via handshake API, this bounds them
        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        if _prof:
            torch.cuda.synchronize()
            _hs0 = perf_counter()
        cache_hits, speculations_tokens, logits_q = self.target_draft_handshake(
            seqs_copy)
        if _prof:
            torch.cuda.synchronize()
            _hs1 = perf_counter()
            print(f"[PROFILE target] handshake_detail={(_hs1-_hs0)*1000:.2f}ms", flush=True)

        # The first column of the final speculation tensor is the recovery tokens
        recovery_tokens_tensor = torch.tensor(
            [seq.recovery_token_id for seq in seqs], dtype=torch.int64, device=self.config.device)
        # Assert that no recovery tokens are None
        assert not torch.any(recovery_tokens_tensor == -
                             1), f"Found None recovery tokens in tensor: {recovery_tokens_tensor}"
        assert recovery_tokens_tensor.numel() > 0, "Recovery tokens tensor is empty"
        speculations = torch.cat(
            [recovery_tokens_tensor.unsqueeze(1), speculations_tokens], dim=1)

        # Update seqs_copy with all speculated tokens for the verify step to pass through the target model
        for i, seq in enumerate(seqs_copy):
            # we added rec token first thing
            seq.token_ids.extend(speculations_tokens[i].tolist())
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = seq.token_ids[-1]
            assert len(
                speculations_tokens[i]) == self.config.speculate_k, f"speculations_tokens[i] should have length {self.config.speculate_k}, got {len(speculations_tokens[i])}"
            seq.num_draft_cached_tokens += len(speculations_tokens[i]) + 1

        # speculations is [B, K+1] at this point since we prepending seq.recovery_token_id for
        return seqs_copy, speculations, logits_q, cache_hits

    def speculate(self, seqs: list[Sequence]):
        # t = perf_counter()
        """Generate k speculative tokens using the draft model."""
        seqs_copy = [seq.clone_spec() for seq in seqs]

        batch_size = len(seqs_copy)

        speculations = torch.zeros(
            batch_size, self.config.speculate_k + 1, dtype=torch.int64, device=self.config.device)
        logits_q = []

        # Single batched write to GPU
        recovery_tokens = []
        for i, seq in enumerate(seqs_copy):
            if seq.recovery_token_id is None:
                raise ValueError(f"recovery_token_id is None for seq {i}")
            recovery_tokens.append(seq.recovery_token_id)
            seq.append_token(seq.recovery_token_id)
        speculations[:, 0] = torch.tensor(
            recovery_tokens, dtype=torch.int64, device="cuda")

        for k in range(self.config.speculate_k + 1):
            # Draft model forward pass - emits [B] tokens, True is for draft_return_logits
            token_ids, step_logits_q = self.draft_runner.call(
                "run", seqs_copy, False, True, True)
            # make sure we include this even on last iter since we put K+1 tokens thru draft cache
            for s in seqs_copy:
                s.num_draft_cached_tokens += 1

            if k == self.config.speculate_k:
                break  # this extra fwd also

            logits_q.append(step_logits_q)

            for i, (seq, token_id) in enumerate(zip(seqs_copy, token_ids)):
                seq.append_token(token_id)

            # Single batched write to GPU
            speculations[:, k + 1] = torch.tensor(
                token_ids, dtype=torch.int64, device="cuda")

        logits_q = torch.stack(logits_q, dim=1)  # [B, K, V]

        return seqs_copy, speculations, logits_q

    def verify(self, seqs_copy: list[Sequence], speculations: torch.Tensor, logits_q: torch.Tensor, cache_hits: torch.Tensor | None = None):
        """Verify speculative tokens using the target model."""
        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        batch_size = len(seqs_copy)

        if _prof:
            torch.cuda.synchronize()
            _vt0 = perf_counter()

        result = self.model_runner.call(
            "run", seqs_copy, False, False, True)

        if _prof:
            torch.cuda.synchronize()
            _vt1 = perf_counter()
        
        if self.config.use_eagle:
            logits_p_flat, eagle_acts_flat = result
        else:
            logits_p_flat = result

        for s in seqs_copy:  # was debuge, but is correct 
            s.num_cached_tokens += self.config.speculate_k + 1

        sleep_for_small_target_debug(self.config)

        logits_p = logits_p_flat.view(
            batch_size, self.config.speculate_k + 1, -1)  # [b, k+1, v]

        # Build per-seq temps for target verify and draft q respectively.
        temps_target = []
        temps_draft = []
        for seq in seqs_copy:
            # If not in async mode, temps remain tied to seq.temperature for both p and q
            if not self.config.draft_async:
                temps_target.append(seq.temperature)
                temps_draft.append(seq.temperature)
                continue

            # Target temperature (verify sampling)
            if getattr(seq, 'target_async_temperature', None) is not None:
                tt = seq.target_async_temperature
            elif self.config.target_async_temp is not None:
                tt = self.config.target_async_temp
            else:
                tt = seq.temperature
            temps_target.append(tt)

            # Draft temperature (q in ratio)
            if getattr(seq, 'draft_async_temperature', None) is not None:
                dt = seq.draft_async_temperature
            elif self.config.draft_async_temp is not None:
                dt = self.config.draft_async_temp
            else:
                dt = seq.temperature
            temps_draft.append(dt)

        temperatures_target = torch.tensor(
            temps_target, dtype=torch.float32, device=self.config.device)
        temperatures_draft = torch.tensor(
            temps_draft, dtype=torch.float32, device=self.config.device)

        # TODO: this will need to work with fan_out_list/fan_out_list_miss? 
        if self.config.sampler_x is not None:
            assert self.config.fan_out_list == self.config.fan_out_list_miss == [self.config.async_fan_out] * (self.config.speculate_k + 1), "fan_out_list and fan_out_list_miss must be the same if sampler_x is provided"
        new_suffixes, recovery_tokens = verify(
            logits_p,
            logits_q,
            speculations,
            temperatures_target,
            temperatures_draft,
            self.config,
            cache_hits=cache_hits,
        )

        if _prof:
            torch.cuda.synchronize()
            _vt2 = perf_counter()
            print(f"[PROFILE verify] target_fwd={(_vt1-_vt0)*1000:.2f}ms verify_compute={(_vt2-_vt1)*1000:.2f}ms", flush=True)

        # Debug: print recovery tokens detokenized
        if __debug__ and recovery_tokens is not None and len(recovery_tokens) > 0:
            tokenizer = self.tokenizer
            recovery_texts = []
            for token in recovery_tokens:
                try:
                    text = tokenizer.decode([token], skip_special_tokens=False)
                    recovery_texts.append(text)
                except Exception:
                    recovery_texts.append(f"<token_id:{token}>")
            print(f"[verify] recovery tokens: {recovery_texts}", flush=True)
        
        METRICS["accepted_suffix_lens_with_recovery"].extend(
            [len(s) for s in new_suffixes]) 

        # For async mode, also track accepted suffix lengths only for cache hits
        if self.config.draft_async and cache_hits is not None:
            for i, suffix_len in enumerate([len(s) for s in new_suffixes]):
                if cache_hits[i] == 1:  # Cache hit
                    METRICS["accepted_suffix_lens_on_hit"].append(suffix_len)
                else: # cache miss
                    METRICS["accepted_suffix_lens_on_miss"].append(suffix_len)

        # Print mean length of new suffixes for monitoring
        if new_suffixes:
            mean_suffix_len = sum(len(suffix)
                                  for suffix in new_suffixes) / len(new_suffixes)
            if __debug__: print(f"[verify] mean new suffix length: {mean_suffix_len:.2f}", flush=True)

        if self.config.use_eagle:
            eagle_acts = eagle_acts_flat.view(batch_size, self.config.speculate_k + 1, -1)
            return new_suffixes, recovery_tokens, eagle_acts
        
        return new_suffixes, recovery_tokens


    def postprocess_speculate(self, seqs: list[Sequence], new_suffixes, recovery_tokens):
        """Update sequences after speculative decoding verification."""
        # Print average length of new suffixes for monitoring spec decoding performance
        ttl_tokens = sum(len(suffix) for suffix in new_suffixes)
        if new_suffixes and self.config.verbose:
            avg_suffix_len = ttl_tokens / len(new_suffixes)
            # for 0.6b-8b it's ~4 tok/s step!!
            print(
                f"avg tokens per spec step: {avg_suffix_len:.2f}", flush=True)

        # updates seqs, kv caches, statuses, eos, etc.
        self.scheduler.postprocess_speculate(
            seqs, new_suffixes, recovery_tokens)

        if self.config.verbose:
            print(
                f'mean new suffix length: {sum(len(s) for s in new_suffixes) / len(new_suffixes):.2f}', flush=True)
        return new_suffixes, recovery_tokens

    def spec_step(self, seqs: list[Sequence]):
        seqs_copy, speculations, logits_q = self.speculate(
            seqs)

        new_suffixes, recovery_tokens = self.verify(
            seqs_copy, speculations, logits_q)
        # handles BOTH block managers and seq state
        self.postprocess_speculate(seqs, new_suffixes, recovery_tokens)
            
        return sum(len(s) for s in new_suffixes)

    def async_draft_prefill_remote(self, seqs: list[Sequence], eagle_acts: torch.Tensor = None):
        skip_first = 1 if self.config.use_eagle else 0
        
        # 1) build all the prefill payload in one shot
        input_ids, positions, cu_q, cu_k, max_q, max_k, slot_map = \
            prepare_prefill_tensors_from_seqs(
                seqs,
                block_size=self.draft_cfg.kvcache_block_size,
                is_draft=True,
                skip_first_token=skip_first
            )
        
        # Slice activations to match draft input
        if eagle_acts is not None:
            sliced_acts = []
            offset = 0
            for seq in seqs:
                seq_len = seq.num_prompt_tokens
                sliced_acts.append(eagle_acts[offset:offset + seq_len - 1])
                offset += seq_len
            eagle_acts = torch.cat(sliced_acts, dim=0)
            assert eagle_acts.shape[0] == input_ids.shape[0], \
                f"Activation length {eagle_acts.shape[0]} != input_ids length {input_ids.shape[0]}"
        
        # 2) pad draft_block_table → block_tables
        max_blocks = (self.draft_cfg.max_model_len +
                      self.draft_cfg.kvcache_block_size - 1)//self.draft_cfg.kvcache_block_size
        block_tables = torch.tensor(
            [s.draft_block_table + [-1] *
                (max_blocks - len(s.draft_block_table)) for s in seqs],
            dtype=torch.int32, device=self.draft_cfg.device,
        )
        
        # 3) send cmd=1
        cmd = torch.tensor([1], dtype=torch.int64, device=self.config.device)
        dist.send(cmd, dst=self.num_tp_gpus, group=self.model_runner.async_pg)

        # 4) send metadata for tensor reconstruction
        metadata = torch.tensor([
            input_ids.size(0),
            slot_map.size(0),
            max_q,
            max_k,
            len(seqs),  # batch_size
        ], dtype=torch.int64, device=self.config.device)
        dist.send(metadata, dst=self.num_tp_gpus,
                  group=self.model_runner.async_pg)

        # 5) send each tensor in a fixed order
        for t in (input_ids, positions, cu_q, cu_k, slot_map, block_tables):
            dist.send(t, dst=self.num_tp_gpus,
                      group=self.model_runner.async_pg)
        
        # 6) send eagle_acts if use_eagle
        if self.config.use_eagle:
            assert eagle_acts is not None, "eagle_acts must be provided when use_eagle is True"
            dist.send(eagle_acts, dst=self.num_tp_gpus, group=self.model_runner.async_pg)

    def spec_prefill(self, seqs: list[Sequence]):
        if self.config.use_eagle:
            # Eagle path: target first (need eagle_acts from target)
            print(f'[spec_prefill] eagle path: target prefill first', flush=True)
            result = self.model_runner.call("run", seqs, True)
            assert isinstance(result, tuple), "result must be a tuple when use_eagle is True"
            assert self.config.draft_async, "Eagle requires async to be true..."
            token_ids, eagle_acts = result
            print(f'[spec_prefill] eagle: got acts, firing draft prefill', flush=True)
            
            offset = 0
            for i, (seq, token_id) in enumerate(zip(seqs, token_ids)):
                seq.recovery_token_id = token_id
                seq_len = seq.num_prompt_tokens
                seq.last_target_hidden_state = eagle_acts[offset + seq_len - 1].clone()
                offset += seq_len
            
            self.async_draft_prefill_remote(seqs, eagle_acts)
        else:
            # Non-eagle path: draft first (enables pipeline overlap)
            print(f'[spec_prefill] non-eagle: draft prefill first', flush=True)
            if self.config.draft_async:
                self.async_draft_prefill_remote(seqs, None)
            else:
                self.draft_runner.call("run", seqs, True)
            
            # Then target prefill
            print(f'[spec_prefill] target prefill', flush=True)
            result = self.model_runner.call("run", seqs, True)
            token_ids = result
            for i, (seq, token_id) in enumerate(zip(seqs, token_ids)):
                seq.recovery_token_id = token_id

        # recovery token will be first token in next fwd, but not yet in kvc of either model
        for i, (seq, token_id) in enumerate(zip(seqs, token_ids)):
            assert seq.recovery_token_id is not None
            seq.num_cached_tokens = seq.num_prompt_tokens
            seq.num_draft_cached_tokens = seq.num_prompt_tokens
        if len(seqs) > 0:
            print(
                f"[PREFILL] seq0 prompt_len={seqs[0].num_prompt_tokens} recovery={seqs[0].recovery_token_id}", flush=True)

    def normal_step(self, seqs: list[Sequence], is_prefill: bool):
        # this includes prepare_decode, set_context, etc
        result = self.model_runner.call("run", seqs, is_prefill)
        if self.config.use_eagle and isinstance(result, tuple):
            token_ids, _ = result
        else:
            token_ids = result
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

    def async_spec_step(self, seqs: list[Sequence]):
        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        if _prof:
            torch.cuda.synchronize()
            _t0 = perf_counter()

        # fire a spec request to hit cache using self.cache_keys, get immediate turnaround
        seqs_copy, speculations, logits_q, cache_hits = self.speculate_async(
            seqs)

        if _prof:
            torch.cuda.synchronize()
            _t1 = perf_counter()

        verify_result = self.verify(
            seqs_copy, speculations, logits_q, cache_hits=cache_hits)

        if _prof:
            torch.cuda.synchronize()
            _t2 = perf_counter()

        if self.config.use_eagle:
            new_suffixes, recovery_tokens, eagle_acts = verify_result

            for i, seq in enumerate(seqs):
                accepted_len = len(new_suffixes[i])
                idx = min(accepted_len - 1, eagle_acts.shape[1] - 1)
                seq.last_target_hidden_state = eagle_acts[i, idx]
        else:
            new_suffixes, recovery_tokens = verify_result

        self.postprocess_speculate(seqs, new_suffixes, recovery_tokens)

        if _prof:
            torch.cuda.synchronize()
            _t3 = perf_counter()
            print(f"[PROFILE target] handshake={(_t1-_t0)*1000:.2f}ms verify={(_t2-_t1)*1000:.2f}ms postprocess={(_t3-_t2)*1000:.2f}ms total={(_t3-_t0)*1000:.2f}ms hits={cache_hits.sum().item()}/{len(cache_hits)} toks={sum(len(s) for s in new_suffixes)}", flush=True)

        return sum(len(s) for s in new_suffixes)

    def step(self):
        t = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        if self.config.speculate:
            if is_prefill:  # assumes this only runs on first prefill, spec loop in else() is usual for specdec
                self.spec_prefill(seqs)
            else:
                ttl_tokens = self.async_spec_step(
                    seqs) if self.config.draft_async else self.spec_step(seqs)
        else:  # normal, non-speculative path
            self.normal_step(seqs, is_prefill)

        time_taken = perf_counter() - t

        if is_prefill:
            METRICS["prefill_total_time"] += time_taken
            METRICS["prefill_total_tokens"] += sum(len(seq) for seq in seqs)
        else:
            METRICS["decode_total_time"] += time_taken
            METRICS["decode_total_tokens"] += ttl_tokens if self.config.speculate else len(
                seqs)

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
            print(
                f"[metrics] Avg Tokens per step (incl recovery): {sum(METRICS['accepted_suffix_lens_with_recovery']) / len(METRICS['accepted_suffix_lens_with_recovery']):.2f}", flush=True)
            num_new_spec_accepts = sum(METRICS['accepted_suffix_lens_with_recovery']) - len(
                METRICS['accepted_suffix_lens_with_recovery'])
            num_spec_steps = len(METRICS['accepted_suffix_lens_with_recovery'])
            avg_acceptance_rate = (
                num_new_spec_accepts / num_spec_steps)/self.config.speculate_k
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
        while not self.is_finished():
            t = perf_counter()
            output = self.step()
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

