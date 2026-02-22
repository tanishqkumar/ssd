import os
import time
import torch
import torch.distributed as dist
import dataclasses

from ssd.engine.model_runner import ModelRunner
from ssd.config import Config
from ssd.utils.context import set_context, reset_context
from ssd.utils.async_helpers.async_spec_helpers import get_forked_recovery_tokens_from_logits, make_glue_decode_input_ids
from ssd.utils.async_helpers.nccl_pack import recv_int64

ttl = 0
ttl_hit = 0

class DraftRunner(ModelRunner):
    
    @classmethod
    def create_draft_config(cls, cfg: Config) -> Config:
        """Create a draft config from the main config without instantiating DraftRunner."""
        draft_cfg = dataclasses.replace(
            cfg,
            model=cfg.draft,
            gpu_memory_utilization = (0.75 if not cfg.draft_async else 0.8), # REMAINING SPACE if not draft_async
            tokenizer_path=cfg.model if cfg.use_eagle else None,
            d_model_target=cfg.hf_config.hidden_size if cfg.use_eagle and cfg.hf_config else None,
            enforce_eager=cfg.enforce_eager,
        )
        return draft_cfg

    def __init__(self, cfg: Config, rank: int = 0, init_q = None):
        self.draft_cfg = self.create_draft_config(cfg)
        self.is_draft = True # this is is_draft, use self.config.draft for the draft model path 
        self.prev_num_tokens = None
        super().__init__(self.draft_cfg, rank=rank, event=None, is_draft=True, num_tp_gpus=1, init_q=init_q)
        
        if self.config.use_eagle:
            assert self.config.jit_speculate, \
                "EAGLE requires jit_speculate=True (cache misses need draft activations)"

        if self.is_draft and self.draft_async:
            self._reset_tree_cache_tensors()
            self._init_prealloc_buffers()
            self._draft_step_times = []
            print(f'DraftRunner set up, starting draft_loop', flush=True)
            self.draft_loop()

    
    # todo:
    # - normally we do a prefill on draft with target activations after we get a verification outcome 
    # - since we wait for the verification outcome to speculate on top of in ordinary (sync) spec
    # - but here (async) when we glue decode the spec we just sent back we obv don't have its target activations yet (since target by defn hasn't verified) so we have to use our own 
    # - where "our own" is stored in the tree cache for each spec rollout, where the glue decode is the tokens we just sent back so they must be in the tree cache
    # - thus, all token fwds are self conditioned in [glue, tree] besides the first token (which is target conditioned input rec token after a verify, with acts from target)
    # - this first token act from target should be sent over nccl along with each cache request, and should correpond to target preact (3 * d_model_target) that led to the logits from which rec token was sampled 
    # - in the first iter, when the token sampled from target sequence to prefill is set as the first rec token, and we make a dummy cache request that always ends in a miss, 
        # eg. with req key ("I", -2, 0) in "how can I" to the draft after target/draft prefill, how should we handle given we sent g_can (which would go with e_I in fwd) target -> draft in fwd? should we send dummy act and not use? 
    def draft_async_prefill(self):
        assert self.draft_async and self.is_draft

        # 1) Receive metadata then individual tensors
        # First recv metadata to learn sizes
        metadata = torch.zeros(5, dtype=torch.int64, device=self.device)
        dist.recv(metadata, src=0, group=self.async_pg)
        total_new_tokens, batch_size, max_blocks, use_eagle, eagle_act_dim = metadata.tolist()
        if use_eagle:
            assert eagle_act_dim == 3 * self.config.d_model_target, (
                f"EAGLE activation dimension {eagle_act_dim} does not match expected dimension 3 * {self.config.d_model_target}"
            )

        # 2) receive fused int64 payload (input_ids + num_tokens + draft_block_table)
        fused_total = total_new_tokens + batch_size + batch_size * max_blocks
        fused = recv_int64(self.async_pg, src=0, total_length=fused_total, device=self.device)
        off = 0
        input_ids = fused[off:off + total_new_tokens]; off += total_new_tokens
        num_tokens = fused[off:off + batch_size]; off += batch_size
        draft_block_table = fused[off:off + batch_size * max_blocks].view(batch_size, max_blocks).to(torch.int32); off += batch_size * max_blocks
        assert off == fused_total

        eagle_acts = None
        if use_eagle:
            eagle_acts = torch.zeros(
                total_new_tokens, eagle_act_dim, dtype=self.hf_config.torch_dtype, device=self.device,
            )
            dist.recv(eagle_acts, src=0, group=self.async_pg)

        prefill_ctxt = self.prepare_prefill_ctxt(num_tokens, draft_block_table)

        # 5) set up context exactly like prepare_prefill() does:
        set_context(
            is_prefill=True,
            cu_seqlens_q=prefill_ctxt["cu_seqlens_q"],
            cu_seqlens_k=prefill_ctxt["cu_seqlens_k"],
            max_seqlen_q=prefill_ctxt["max_seqlen_q"],
            max_seqlen_k=prefill_ctxt["max_seqlen_k"],
            slot_mapping=prefill_ctxt["slot_map"],
            context_lens=None,
        ) # , block_tables=block_tables, commenting this out essentially removes prefix caching

        # 6) run the draft model in prefill mode
        positions = prefill_ctxt["positions"]
        if self.config.use_eagle:
            self.run_model(input_ids, positions, is_prefill=True, last_only=True, hidden_states=eagle_acts)
        else:
            self.run_model(input_ids, positions, is_prefill=True, last_only=True, hidden_states=eagle_acts)

        # 7) clean up
        reset_context()

    def _reset_tree_cache_tensors(self):
        """Reset tensor-backed tree cache to empty."""
        # initialize as empty keys on correct device; tokens/logits set to None until first populate
        self.tree_cache_keys = torch.zeros(
            (0, 3), dtype=torch.int64, device=self.device)
        self.tree_cache_tokens = None
        self.tree_cache_logits = None
        self.tree_cache_activations = None

    def _init_prealloc_buffers(self):
        # PERFORMANCE: pre-allocate constant tensors used every draft step to avoid repeated CUDA mallocs
        K, MQ_LEN = self.config.speculate_k, self.config.MQ_LEN
        d = self.device
        self._step_pos_offsets = torch.arange(K, device=d, dtype=torch.int64)[:, None] * MQ_LEN
        self._step_rope_offsets = torch.arange(K, device=d, dtype=torch.int64)[:, None]
        self._fan_idx_hit = torch.arange(K + 1, device=d, dtype=torch.int64).repeat_interleave(self.config.fan_out_t)
        self._fan_idx_miss = torch.arange(K + 1, device=d, dtype=torch.int64).repeat_interleave(self.config.fan_out_t_miss)
        self._arange_mq = torch.arange(MQ_LEN, device=d, dtype=torch.int64)
        self._arange_kp1 = torch.arange(K + 1, device=d, dtype=torch.int64)
        self._arange_2kp1 = torch.arange(2 * K + 1, device=d, dtype=torch.int64)

    def jit_speculate(self, 
                      request_keys: torch.Tensor, 
                      num_tokens: torch.Tensor, 
                      out_logits: torch.Tensor, 
                      out_tokens: torch.Tensor, 
                      temperatures: torch.Tensor, 
                      draft_block_tables: torch.Tensor,
                      target_recovery_activations: torch.Tensor = None):
        
        input_ids = request_keys[:, -1]
        pos_offset = -1 if self.config.use_eagle else 0
        positions = num_tokens - 1 + pos_offset # want to write rec token at post N-1 since [0, ..., N-2] filled by prefill 
        context_lens = num_tokens + pos_offset # N+1
        # Calculate slot mapping vectorized
        block_idx = positions // self.block_size
        pos_in_block = positions % self.block_size
        batch_indices = torch.arange(input_ids.shape[0], device=self.device)
        slot_map = draft_block_tables[batch_indices, block_idx] * self.block_size + pos_in_block

        hidden_states = None
        spec_activations = None
        
        if self.config.use_eagle:
            assert target_recovery_activations is not None
            hidden_states = self.model.fc(target_recovery_activations.to(self.model.fc.weight.dtype))
            spec_activations = torch.empty(
                input_ids.shape[0], self.config.speculate_k,
                self.hf_config.hidden_size,
                dtype=self.hf_config.torch_dtype, device=self.device)

        for i in range(self.config.speculate_k): # we're going to glue after this anyways, and by sending the spec request target has verified we have K more slots left in our last page 
            set_context(
                is_prefill=False,
                slot_mapping=slot_map,
                context_lens=context_lens.to(torch.int32),
                block_tables=draft_block_tables,
                is_jit=True,
            )
            
            if self.config.use_eagle:
                logits, prenorm = self.run_model(input_ids, positions, is_prefill=False, last_only=True, hidden_states=hidden_states)
                spec_activations[:, i] = prenorm
                hidden_states = prenorm
            else:
                logits = self.run_model(input_ids, positions, is_prefill=False, last_only=True)
            
            out_logits[:, i, :] = logits
            reset_context()
            next_tokens = self.sampler(logits, temperatures, is_tree=True)
            out_tokens[:, i] = next_tokens
            
            # Update for next iteration
            input_ids = next_tokens
            positions = positions + 1
            context_lens = context_lens + 1
            # Update slot mapping for next position
            block_idx = positions // self.block_size
            pos_in_block = positions % self.block_size
            slot_map = draft_block_tables[batch_indices, block_idx] * self.block_size + pos_in_block

        return spec_activations

    def hit_cache_and_respond(self, request_keys, B, K, num_tokens, temperatures, draft_block_tables, target_recovery_activations=None):
        """Hits the cache (tensor-backed) and returns tensors to respond to the spec request."""
        global ttl, ttl_hit
        # Draft model now returns full target vocab size logits (after d2t expansion)
        V = self.hf_config.vocab_size

        # Init miss slots with valid random logits so token IDs are in-vocab (fixes B>1 crash)
        out_logits = torch.empty((B, K, V), dtype=self.hf_config.torch_dtype, device=self.device).uniform_()
        out_tokens = out_logits.argmax(dim=-1)
        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)

        assert request_keys.shape == (B, 3), f"ERROR in hit_cache_and_respond: request_keys should be (B, 3), got {request_keys.shape}"
        
        hidden_size = self.hf_config.hidden_size
        out_activations = torch.zeros(
            B, K, hidden_size,
            dtype=self.hf_config.torch_dtype, device=self.device
        ) if self.config.use_eagle else None
        
        # Statistics
        ttl += int(B)
        
        if self.config.verbose:
            print(f"[hit_cache_and_respond] Request keys: {request_keys}", flush=True)
            for i in range(B):
                rec_token = request_keys[i, 2].item()
                rec_text = self.tokenizer.decode([rec_token])
                print(f"  Req {i}: token={rec_token} ('{rec_text}')", flush=True)
        
        if self.tree_cache_keys.numel() > 0:
            # Vectorized membership against tensor cache
            eq = (request_keys.unsqueeze(1) == self.tree_cache_keys.unsqueeze(0))  # [B,T,3]
            match = torch.all(eq, dim=2)  # [B,T]
            cache_hits = match.any(dim=1)  # [B]
            ttl_hit += int(cache_hits.sum().item())
            
            if self.config.verbose:
                print(f"[hit_cache_and_respond] Cache hits: {cache_hits.sum().item()}/{B}", flush=True)
                print(f"[hit_cache_and_respond] Cache: {self.tree_cache_keys.shape[0]} entries", flush=True)
                
                # Build set of hit cache indices for marking
                hit_indices = set()
                if cache_hits.any():
                    idx = match.float().argmax(dim=1).to(torch.int64)
                    for i in range(B):
                        if cache_hits[i]:
                            hit_indices.add(idx[i].item())
                
                # Print cache entries with hit markers
                for i, key in enumerate(self.tree_cache_keys):
                    seq_id, k_idx, rec_token = key.tolist()
                    rec_text = self.tokenizer.decode([rec_token])
                    hit_marker = "[HIT]" if i in hit_indices else ""
                    print(f"    [{i}]: key=({seq_id}, {k_idx}, {rec_token}) -> value=('{rec_text}') {hit_marker}", flush=True)
            
            # Fill hits
            if (cache_hits.any() and not self.config.jit_speculate) or (cache_hits.all() and self.config.jit_speculate):
                # print(f'[hit_cache_and_respond] got all cache hits, using cached logits and tokens', flush=True)
                # [B], arbitrary if no match but masked out
                idx = match.float().argmax(dim=1).to(torch.int64)
                sel = cache_hits
                # tokens [T,K]
                out_tokens[sel] = self.tree_cache_tokens[idx[sel]]
                # logits [T,K+1,V]
                out_logits[sel] = self.tree_cache_logits[idx[sel]]
                if self.config.use_eagle:
                    out_activations[sel] = self.tree_cache_activations[idx[sel]]
            elif self.config.jit_speculate: 
                # print(f'[hit_cache_and_respond] found a cache miss, running jit speculate', flush=True)
                if self.config.verbose:
                    print(f"[hit_cache_and_respond] Running JIT speculate for cache misses", flush=True)
                jit_acts = self.jit_speculate(
                    request_keys, 
                    num_tokens, 
                    out_logits, 
                    out_tokens,
                    temperatures,
                    draft_block_tables,
                    target_recovery_activations
                    ) # write into out_logits, out_tokens
                if self.config.use_eagle:
                    out_activations = jit_acts
        elif self.config.jit_speculate:
            # Cache is empty (first iteration), must JIT all
            if self.config.verbose:
                print(f"[hit_cache_and_respond] Cache empty, running JIT speculate for all", flush=True)
            jit_acts = self.jit_speculate(
                request_keys, 
                num_tokens, 
                out_logits, 
                out_tokens,
                temperatures,
                draft_block_tables,
                target_recovery_activations
                )
            if self.config.use_eagle:
                out_activations = jit_acts
            
        rec_toks = request_keys[:, 2]
        
        return out_tokens, out_logits, make_glue_decode_input_ids(out_tokens, rec_toks), cache_hits, out_activations

    def _service_spec_request(self):
        """Receives a speculation request, serves it from cache, and sends results back in a single response."""
        meta = self.recv_tensor((3,), torch.int64)
        B, K, F = meta.tolist()

        # Receive all request payload in one fused int64 burst (includes temperatures encoded as int64)
        max_blocks = self.config.max_blocks
        fused_total = (3 * B) + B + (B * max_blocks) + B  # +B for temps_as_int64
        fused_req = recv_int64(self.async_pg, src=0,
                               total_length=fused_total, device=self.device)
        off = 0
        cache_keys = fused_req[off:off + (3 * B)].view(B, 3)
        off += 3 * B
        seq_ids = cache_keys[:, 0]
        num_tokens = fused_req[off:off + B].to(torch.int64)
        off += B
        draft_block_tables = fused_req[off:off + B *
                                       max_blocks].view(B, max_blocks).to(torch.int32)
        off += B * max_blocks
        temps_as_int64 = fused_req[off:off + B]
        off += B
        assert off == fused_total
        temperatures = temps_as_int64.to(torch.int32).view(torch.float32)

        target_recovery_activations = torch.zeros(
            B, 3 * self.config.d_model_target, dtype=self.hf_config.torch_dtype, device=self.device
        ) if self.config.use_eagle else None

        extend_counts = None
        extend_eagle_acts = None
        extend_token_ids = None

        if self.config.use_eagle:
            dist.recv(target_recovery_activations, src=0, group=self.async_pg)

            # Receive extend data for fused glue decode
            act_dim = 3 * self.config.d_model_target
            extend_counts = torch.zeros(B, dtype=torch.int64, device=self.device)
            extend_eagle_acts = torch.zeros(B, K, act_dim, dtype=self.hf_config.torch_dtype, device=self.device)
            extend_token_ids = torch.zeros(B, K, dtype=torch.int64, device=self.device)
            dist.recv(extend_counts, src=0, group=self.async_pg)
            dist.recv(extend_eagle_acts, src=0, group=self.async_pg)
            dist.recv(extend_token_ids, src=0, group=self.async_pg)

            if self.config.verbose:
                recovery_tokens_target = cache_keys[:, 2].clone()
                print(f"\n{'='*80}", flush=True)
                print(f"[CACHE REQUEST] Batch size: {B}, Spec depth: {K}", flush=True)
                for i in range(B):
                    seq_id = cache_keys[i, 0].item()
                    keep_idx = cache_keys[i, 1].item()
                    rec_token_target = recovery_tokens_target[i].item()
                    rec_token_text = self.tokenizer.decode([rec_token_target])
                    n_ext = extend_counts[i].item()
                    print(f"  Seq {seq_id}: keep_idx={keep_idx}, recovery_token={rec_token_target} ('{rec_token_text}'), n_ext={n_ext}", flush=True)
                print(f"{'='*80}\n", flush=True)

        out_tokens, out_logits, glue_decode_input_ids, cache_hits, out_activations = self.hit_cache_and_respond(
            cache_keys, B, K, num_tokens, temperatures, draft_block_tables, target_recovery_activations)

        if self.config.verbose:
            print(f"[CACHE RESPONSE]", flush=True)
            for i in range(B):
                hit_status = "HIT" if cache_hits[i].item() == 1 else "MISS"
                print(f"  Seq {cache_keys[i, 0].item()}: {hit_status}", flush=True)
                if cache_hits[i].item() == 1 or self.config.jit_speculate:
                    tokens_list = out_tokens[i, :K].tolist()
                    tokens_text = [self.tokenizer.decode([t]) for t in tokens_list]
                    print(f"    Tokens: {tokens_list}", flush=True)
                    print(f"    Detokenized: {tokens_text}", flush=True)
            print(f"", flush=True)

        fused_response = torch.cat([cache_hits.reshape(-1), out_tokens.reshape(-1).to(torch.int64)])
        dist.send(fused_response, dst=0, group=self.async_pg)
        dist.send(out_logits[:, :K, :].contiguous(), dst=0, group=self.async_pg)

        partial_tree_decode_args = {
            "num_tokens": num_tokens,
            "seq_ids": seq_ids,
            "temperatures": temperatures,
            "dbt": draft_block_tables,
            "cache_hits": cache_hits,
            "returned_tokens": out_tokens,
            "target_recovery_activations": target_recovery_activations,
            "previous_activations": out_activations,
            "extend_counts": extend_counts,
            "extend_eagle_acts": extend_eagle_acts,
            "extend_token_ids": extend_token_ids,
        }

        return glue_decode_input_ids, partial_tree_decode_args

    def prepare_prefill_ctxt(
        self,
        num_tokens: torch.Tensor,  # [B]
        draft_block_table: torch.Tensor,  # [B, max_blocks]
    ) -> dict:
        """
        Prepare context for prefill forward pass.
        """
        B = num_tokens.shape[0]
        total = num_tokens.sum().item()
        cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q[1:] = torch.cumsum(num_tokens, dim=0)
        batch_indices = torch.arange(B, device=self.device, dtype=torch.int64).repeat_interleave(num_tokens)
        positions = torch.arange(total, device=self.device, dtype=torch.int64) - cu_seqlens_q[:-1].to(torch.int64).repeat_interleave(num_tokens)
        max_seqlen_q = num_tokens.max().item()

        # Calculate block indices and offsets for ALL positions
        block_indices = (positions // self.block_size).to(torch.int64)
        offsets = (positions % self.block_size).to(torch.int32)

        # Get block IDs for each position from dbt
        block_ids = draft_block_table[batch_indices, block_indices]

        # Calculate slot_map for each position
        slot_map = (block_ids * self.block_size + offsets).to(torch.int32)

        return {
            "positions": positions,
            "slot_map": slot_map,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_q.clone(),
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_q,
        }

    
    def prepare_glue_decode_ctxt(self, num_tokens, input_ids, dbt, B):
        K = self.config.speculate_k
        pos_offset = -1 if self.config.use_eagle else 0
        positions_start = (num_tokens - 1 + pos_offset).unsqueeze(-1)
        positions_grid = positions_start + self._arange_kp1

        # Calculate block indices and offsets for ALL positions
        block_indices = (positions_grid // self.block_size).to(torch.int64)
        offsets = (positions_grid % self.block_size).to(torch.int32)

        # Get block IDs for each position from dbt
        B_expanded = torch.arange(B, device=self.device).unsqueeze(-1).expand(-1, K + 1)
        blk_ids = dbt[B_expanded, block_indices]

        # Calculate slot_map for each position
        slot_map_grid = blk_ids * self.block_size + offsets

        # Flattened tensors for varlen decode
        positions_flat = positions_grid.reshape(-1).to(torch.int64)
        slot_map_flat = slot_map_grid.reshape(-1).to(torch.int32)

        context_lens = (num_tokens + pos_offset + K).to(torch.int32)
        seqlen_q = torch.full((B,), K + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)

        return {
            "input_ids": input_ids,
            "positions": positions_flat,
            "slot_map": slot_map_flat,
            "cu_seqlens_q": cu_seqlens_q,
            "max_seqlen_q": K + 1,
            "context_lens": context_lens,
            "block_tables": dbt,
        }

    def prepare_glue_decode_ctxt_eagle(self, num_tokens, fused_ids, fused_hs, extend_counts, seqlens_q, cu_seqlens_q, dbt, B):
        """Prepare context for EAGLE glue decode with FA varlen causal.

        Tokens packed contiguously: [ext_0..ext_{n0-1}, rec_0, spec_0..spec_{K-1}, ext_1..., ...]
        No padding within sequences. cu_seqlens_q has variable per-seq lengths.
        """
        K = self.config.speculate_k
        total_real = int(cu_seqlens_q[-1].item())

        # Per-token batch index and local offset within each seq
        batch_idx = torch.repeat_interleave(torch.arange(B, device=self.device), seqlens_q)  # [total_real]
        local_off = torch.arange(total_real, device=self.device) - cu_seqlens_q[:-1].long().repeat_interleave(seqlens_q)

        # Positions: extend starts at num_tokens-2-n_ext, then rec, then spec
        # base_pos[b] = num_tokens[b] - 2 - extend_counts[b] (position of first extend token)
        base_pos = (num_tokens - 2 - extend_counts).long()  # [B]
        positions = (base_pos[batch_idx] + local_off).to(torch.int64)

        # Context lens: last token (spec K-1) at pos num_tokens-2+K, cache has 0..num_tokens-2+K
        context_lens = (num_tokens - 1 + K).to(torch.int32)

        # Slot mapping
        block_idx = (positions // self.block_size).clamp(0, dbt.shape[1] - 1).to(torch.int64)
        block_off = (positions % self.block_size).to(torch.int32)
        blk_ids = dbt[batch_idx, block_idx]
        slot_map = (blk_ids * self.block_size + block_off).to(torch.int32)

        return {
            "input_ids": fused_ids,
            "positions": positions,
            "slot_map": slot_map,
            "hidden_states": fused_hs,
            "cu_seqlens_q": cu_seqlens_q,
            "max_seqlen_q": 2 * K + 1,
            "context_lens": context_lens,
            "block_tables": dbt,
        }

    def _construct_tree_decode_args(self, partial_tree_decode_args, rec_flat, dbt):
        # tree decode needs (input_ids, positions) that are [N], wrapper plan handles batch size of attn computation 
        # rec_flat is [N]
        
        B = dbt.shape[0]
        K = self.config.speculate_k
        F = self.config.async_fan_out
        N = rec_flat.shape[0]
        cache_hits = partial_tree_decode_args["cache_hits"]

        if __debug__:
            assert N == B*self.config.MQ_LEN, f"ERROR in _construct_tree_decode_args: N should be B*self.config.MQ_LEN={B*self.config.MQ_LEN}, got {N}"

        b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[:, None].expand(B, self.config.MQ_LEN).flatten()
        fkp1_flat = self._arange_mq.repeat(B)
        j_idx_flat = torch.cat([self._fan_idx_hit if hit else self._fan_idx_miss for hit in cache_hits])
        metadata = torch.tensor([B, K, F, N], dtype=torch.int64, device=self.device)

        seq_ids = partial_tree_decode_args["seq_ids"]
        seq_ids_expanded = seq_ids[b_flat]
        pos_offset = -1 if self.config.use_eagle else 0
        positions = (partial_tree_decode_args["num_tokens"][b_flat] - 1 + pos_offset) + (K + 1) + fkp1_flat
        rope_positions = (partial_tree_decode_args["num_tokens"][b_flat] - 1 + pos_offset) + j_idx_flat + 1
        temperatures = partial_tree_decode_args["temperatures"][b_flat]

        tree_decode_args = {
            "metadata": metadata,
            "input_ids": rec_flat,  # [N]
            "positions": positions,  # [N]
            "rope_positions": rope_positions, # [N], these are to be passed into model fwd 
            # the dbt is now [B, M] in the seq fan out codebase
            "block_tables": dbt,
            "temps": temperatures,  # [N]
            "rec_flat": rec_flat,  # [N]
            "seq_ids_expanded": seq_ids_expanded,  # [N]
            "cache_hits": cache_hits,  # [B] # we also want returned_tokens which is [B, K]
        }

        return tree_decode_args

    def _build_tree_batch(self, partial_tree_decode_args, glue_decode_input_ids):
        if self.config.verbose:
            print(f'about to build tree batch')
        K = self.config.speculate_k
        dbt = partial_tree_decode_args["dbt"]
        cache_hits = partial_tree_decode_args["cache_hits"]
        cache_hits_list = cache_hits.tolist()
        pos_offset = -1 if self.config.use_eagle else 0

        if self.config.use_eagle:
            B = partial_tree_decode_args["num_tokens"].shape[0]
            extend_counts = partial_tree_decode_args.get("extend_counts")
            if extend_counts is None:
                extend_counts = torch.zeros(B, dtype=torch.int64, device=self.device)
            extend_eagle_acts_batch = partial_tree_decode_args.get("extend_eagle_acts")
            extend_token_ids_batch = partial_tree_decode_args.get("extend_token_ids")
            target_acts = partial_tree_decode_args["target_recovery_activations"]
            prev_acts = partial_tree_decode_args["previous_activations"]
            hidden_size = self.hf_config.hidden_size
            fc_dtype = self.model.fc.weight.dtype

            gd_view = glue_decode_input_ids.view(B, K + 1)
            rec_tok_ids = gd_view[:, 0]
            spec_tok_ids = gd_view[:, 1:]

            # Variable per-seq lengths: n_ext[b] + K + 1
            seqlens_q = (extend_counts + K + 1).to(torch.int32)
            cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
            cu_seqlens_q[1:] = torch.cumsum(seqlens_q, 0)
            total_real = int(cu_seqlens_q[-1].item())

            # Build packed fused_ids and fused_hs (no padding, no for loops)
            fused_ids = torch.zeros(total_real, dtype=torch.int64, device=self.device)
            fused_hs = torch.zeros(total_real, hidden_size, dtype=self.hf_config.torch_dtype, device=self.device)

            # Per-token batch index and local offset
            batch_idx = torch.repeat_interleave(torch.arange(B, device=self.device), seqlens_q)
            local_off = torch.arange(total_real, device=self.device) - cu_seqlens_q[:-1].long().repeat_interleave(seqlens_q)
            n_ext = extend_counts.long()  # [B]
            n_ext_per_tok = n_ext[batch_idx]  # [total_real]

            # Classify each token: extend (local < n_ext), rec (local == n_ext), spec (local > n_ext)
            is_extend = local_off < n_ext_per_tok
            is_rec = local_off == n_ext_per_tok
            is_spec = local_off > n_ext_per_tok

            # Extend + rec tokens: batch fc into single call
            is_target_conditioned = is_extend | is_rec
            tc_b = batch_idx[is_target_conditioned]
            tc_local = local_off[is_target_conditioned]
            tc_n_ext = n_ext_per_tok[is_target_conditioned]

            # Gather target acts: extend uses extend_eagle_acts_batch[b,j], rec uses target_acts[b]
            tc_is_ext = tc_local < tc_n_ext
            tc_acts = torch.empty(tc_b.size(0), target_acts.size(1), dtype=fc_dtype, device=self.device)
            if tc_is_ext.any() and extend_eagle_acts_batch is not None:
                ext_b = tc_b[tc_is_ext]
                ext_j = tc_local[tc_is_ext]
                tc_acts[tc_is_ext] = extend_eagle_acts_batch[ext_b, ext_j].to(fc_dtype)
                fused_ids[is_extend] = extend_token_ids_batch[ext_b, ext_j]
            tc_acts[~tc_is_ext] = target_acts[tc_b[~tc_is_ext]].to(fc_dtype)
            fused_ids[is_rec] = rec_tok_ids[batch_idx[is_rec]]

            # Single batched fc call
            fused_hs[is_target_conditioned] = self.model.fc(tc_acts)

            # Spec tokens: ids from spec_tok_ids, hs from prev_acts (self-conditioned, no fc)
            spec_j = local_off[is_spec] - n_ext_per_tok[is_spec] - 1  # 0..K-1
            fused_ids[is_spec] = spec_tok_ids[batch_idx[is_spec], spec_j]
            fused_hs[is_spec] = prev_acts[batch_idx[is_spec], spec_j]

            glue_decode_ctxt = self.prepare_glue_decode_ctxt_eagle(
                num_tokens=partial_tree_decode_args["num_tokens"],
                fused_ids=fused_ids, fused_hs=fused_hs,
                extend_counts=extend_counts, seqlens_q=seqlens_q,
                cu_seqlens_q=cu_seqlens_q, dbt=dbt, B=B,
            )
        else:
            # Non-EAGLE: K+1 per seq, uses verify CG path
            B = glue_decode_input_ids.shape[0] // (K + 1)
            assert B == partial_tree_decode_args["num_tokens"].shape[0]
            glue_decode_ctxt = self.prepare_glue_decode_ctxt(
                num_tokens=partial_tree_decode_args["num_tokens"],
                input_ids=glue_decode_input_ids,
                dbt=dbt, B=B,
            )

        # Pre-compute tree decode args (overlap CPU with GPU)
        _pre_b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[:, None].expand(B, self.config.MQ_LEN).flatten()
        _pre_fkp1_flat = self._arange_mq.repeat(B)
        _pre_j_idx_flat = torch.cat([self._fan_idx_hit if int(h) else self._fan_idx_miss for h in cache_hits_list])
        N_pre = _pre_b_flat.shape[0]
        _pre_metadata_ints = (B, K, self.config.async_fan_out, N_pre)
        _pre_seq_ids_expanded = partial_tree_decode_args["seq_ids"][_pre_b_flat]
        _pre_positions = (partial_tree_decode_args["num_tokens"][_pre_b_flat] - 1 + pos_offset) + (K + 1) + _pre_fkp1_flat
        _pre_rope_positions = (partial_tree_decode_args["num_tokens"][_pre_b_flat] - 1 + pos_offset) + _pre_j_idx_flat + 1
        _pre_temperatures = partial_tree_decode_args["temperatures"][_pre_b_flat]

        # --- Run glue decode forward ---
        set_context(
            is_prefill=False,
            cu_seqlens_q=glue_decode_ctxt["cu_seqlens_q"],
            max_seqlen_q=glue_decode_ctxt["max_seqlen_q"],
            slot_mapping=glue_decode_ctxt["slot_map"],
            context_lens=glue_decode_ctxt["context_lens"],
            block_tables=glue_decode_ctxt["block_tables"],
        )

        glue_prenorm = None
        if self.config.use_eagle:
            fused_hs_flat = glue_decode_ctxt["hidden_states"]
            glue_decode_logits_flat, glue_prenorm = self.run_model(
                glue_decode_ctxt["input_ids"], glue_decode_ctxt["positions"],
                is_prefill=False, last_only=False, hidden_states=fused_hs_flat)
        else:
            glue_decode_logits_flat = self.run_model(
                glue_decode_ctxt["input_ids"], glue_decode_ctxt["positions"],
                is_prefill=False, last_only=False)

        reset_context()

        # --- Extract K+1 logits/prenorms at rec+spec positions ---
        if self.config.use_eagle:
            # Packed layout: rec at cu_seqlens_q[b] + n_ext[b], spec follows
            cu_q = glue_decode_ctxt["cu_seqlens_q"]
            rec_offsets = cu_q[:-1].long() + extend_counts.long()  # [B]
            extract_idx = rec_offsets.unsqueeze(1) + self._arange_kp1.unsqueeze(0)  # [B, K+1]
            flat_idx = extract_idx.flatten()
            glue_decode_logits = glue_decode_logits_flat[flat_idx].view(B, K + 1, -1)
            if glue_prenorm is not None:
                glue_prenorm_kp1 = glue_prenorm[flat_idx].view(B, K + 1, -1)
        else:
            glue_decode_logits = glue_decode_logits_flat.view(B, K + 1, -1)
            if glue_prenorm is not None:
                glue_prenorm_kp1 = glue_prenorm.view(B, K + 1, -1)

        # --- Build tree hidden states from K+1 prenorms ---
        tree_hidden_states = None
        if glue_prenorm is not None:
            # Vectorized: for each (b, depth), repeat prenorm by fan_out[depth]
            # fan_out_t[depth] for hits, fan_out_t_miss[depth] for misses
            fan_hit = self.config.fan_out_t  # [K+1]
            fan_miss = self.config.fan_out_t_miss  # [K+1]
            # Per-batch fan_out: [B, K+1]
            per_batch_fan = torch.where(
                cache_hits.bool().unsqueeze(1).expand(B, K + 1),
                fan_hit.unsqueeze(0).expand(B, K + 1),
                fan_miss.unsqueeze(0).expand(B, K + 1),
            )  # [B, K+1]
            reps_flat = per_batch_fan.reshape(-1)  # [B*(K+1)]
            prenorms_flat = glue_prenorm_kp1.reshape(B * (K + 1), -1)  # [B*(K+1), d]
            tree_hidden_states = torch.repeat_interleave(prenorms_flat, reps_flat, dim=0)

        # --- Fork tokens from K+1 logits ---
        # Need [B, K+1] input_ids for forking (rec + spec tokens)
        if self.config.use_eagle:
            gd_for_fork = gd_view  # [B, K+1] already computed above
        else:
            gd_for_fork = glue_decode_input_ids.reshape(B, K + 1)

        forked_rec_tokens = get_forked_recovery_tokens_from_logits(
            self.config,
            glue_decode_logits,
            cache_hits,
            gd_for_fork,
            tokenizer=self.tokenizer,
        ).view(-1)

        tree_decode_args = {
            "metadata_ints": _pre_metadata_ints,
            "input_ids": forked_rec_tokens,
            "positions": _pre_positions,
            "rope_positions": _pre_rope_positions,
            "block_tables": dbt,
            "temps": _pre_temperatures,
            "rec_flat": forked_rec_tokens,
            "seq_ids_expanded": _pre_seq_ids_expanded,
            "cache_hits": cache_hits,
            "cache_hits_list": cache_hits_list,
        }
        tree_decode_args["hidden_states"] = tree_hidden_states
        return tree_decode_args

    @torch.inference_mode()
    def _compute_step_positions_and_slot_maps(self, initial_positions, initial_rope_positions, dbt, B, K, F, N, MQ_LEN):
        # PERFORMANCE: pre-allocated _step_pos_offsets/_step_rope_offsets avoid per-step torch.arange calls
        step_positions = initial_positions[None, :] + self._step_pos_offsets
        step_rope_positions = initial_rope_positions[None, :] + self._step_rope_offsets
        step_context_lens = step_positions.view(K, B, MQ_LEN)[:, :, -1] + 1

        # Precompute slot_maps for all steps: [K, N]
        b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[
            :, None].expand(B, self.config.MQ_LEN).flatten()
        batch_indices = torch.arange(N, device=self.device)
        dbt_expanded = dbt[b_flat]  # [N, M] - constant across steps

        step_offsets = (step_positions % self.block_size).to(torch.int32)  # [K, N]
        step_last_blks = (step_positions // self.block_size).to(torch.int64)  # [K, N]
        step_blk_ids = dbt_expanded[batch_indices[None, :], step_last_blks]  # [K, N]
        step_slot_maps = step_blk_ids * self.block_size + step_offsets  # [K, N]

        return step_positions, step_rope_positions, step_context_lens, step_slot_maps

    def _decode_tree_step(self, depth, current_input_ids, step_rope_positions, step_slot_maps, step_context_lens, dbt, payload, spec_tokens, spec_logits, spec_activations):
        """Execute a single tree decode step."""
        # Use precomputed values for this step
        set_context(
            is_prefill=False,
            slot_mapping=step_slot_maps[depth],
            context_lens=step_context_lens[depth].to(torch.int32),
            block_tables=dbt,
        )

        hidden_states = payload.get("hidden_states")
        if self.config.use_eagle:
            logits, prenorm = self.run_model(current_input_ids, step_rope_positions[depth], is_prefill=False, last_only=False, tree_decode_step=depth, cache_hits=payload["cache_hits"], hidden_states=hidden_states)
            assert spec_activations is not None
            spec_activations[:, depth] = prenorm
            payload["hidden_states"] = prenorm
        else:
            logits = self.run_model(current_input_ids, step_rope_positions[depth], is_prefill=False, last_only=False, tree_decode_step=depth, cache_hits=payload["cache_hits"])
        
        reset_context()
        
        V = self.hf_config.vocab_size  # Draft returns full target vocab size after d2t expansion
        logits_flat = logits.view(-1, V)  # [N, V]
        spec_logits[:, depth, :] = logits_flat
        # Inline greedy: payload["_all_greedy"] checked once in _decode_tree
        next_tokens = logits_flat.argmax(dim=-1) if payload["_all_greedy"] else self.sampler(logits_flat, payload["temps"], is_tree=True)
        spec_tokens[:, depth] = next_tokens
        
        return next_tokens

    def _decode_tree(self, payload):
        """Decodes the speculation tree, checking for interrupts at each step."""

        # setup
        B, K, F, N = payload["metadata_ints"]

        V = self.hf_config.vocab_size  # Draft returns full target vocab size after d2t expansion
        spec_tokens = torch.zeros(
            (N, K), dtype=torch.int64, device=self.device)
        spec_logits = torch.zeros(
            (N, K, V), dtype=self.hf_config.torch_dtype, device=self.device)
        spec_activations = torch.zeros(
            (N, K, self.hf_config.hidden_size),
            dtype=self.hf_config.torch_dtype, device=self.device
        ) if self.config.use_eagle else None

        # Precompute all positions, context_lens, and slot_maps for all K steps
        # PERFORMANCE: no .clone() needed — these are not modified in-place
        initial_positions = payload["positions"]  # [N]
        initial_rope_positions = payload["rope_positions"]  # [N]
        current_input_ids = payload["input_ids"]  # [N], the forked tokens
        dbt = payload["block_tables"]  # [B, M] - constant across steps
        
        # Use compiled function for batch-size independent computations
        _, step_rope_positions, step_context_lens, step_slot_maps = self._compute_step_positions_and_slot_maps(
            initial_positions, initial_rope_positions, dbt, B, K, F, N, self.config.MQ_LEN
        )

        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        payload["_all_greedy"] = bool((payload["temps"] == 0).all())
        for depth in range(K):
            if _prof:
                torch.cuda.synchronize()
                _st = time.perf_counter()
            current_input_ids = self._decode_tree_step(
                depth, current_input_ids, step_rope_positions, step_slot_maps,
                step_context_lens, dbt, payload, spec_tokens, spec_logits, spec_activations
            )
            if _prof:
                torch.cuda.synchronize()
                _et = time.perf_counter()
                print(f"[PROFILE draft] tree_step[{depth}]={(_et-_st)*1000:.2f}ms", flush=True)

        return spec_tokens, spec_logits, spec_activations

    def _populate_tree_cache(self, payload, tokens, logits, cache_hits, activations=None):
        """Populates the tensor-backed tree_cache with the results of the decoding.
        """
        seq_ids_expanded = payload["seq_ids_expanded"].to(torch.int64)
        rec_flat = payload["rec_flat"].to(torch.int64)

        # B = payload["block_tables"].shape[0]
        # k_flat = torch.arange(self.config.speculate_k + 1, device=self.device, dtype=torch.int64)[None, :, None].expand(
        #     B, self.config.speculate_k + 1, self.config.async_fan_out).flatten()

        k_flat = torch.cat([self._fan_idx_hit if hit else self._fan_idx_miss for hit in payload["cache_hits_list"]])

        assert k_flat.shape[0] == payload["block_tables"].shape[0] * self.config.MQ_LEN, f"ERROR in _populate_tree_cache: k_flat should be {payload['block_tables'].shape[0] * self.config.MQ_LEN}, got {k_flat.shape[0]}"
        
        keys = torch.stack([seq_ids_expanded, k_flat, rec_flat], dim=1).contiguous()  # [N,3]

        assert self.tree_cache_keys.numel() == 0
        self.tree_cache_keys = keys
        self.tree_cache_tokens = tokens
        self.tree_cache_logits = logits
        self.tree_cache_activations = activations
        
        # Print cache population details
        if self.config.verbose:
            N = keys.shape[0]
            print(f"\n{'='*80}", flush=True)
            print(f"[CACHE POPULATED] {N} entries", flush=True)
            
            # Show sample entries per sequence
            for seq_id in keys[:, 0].unique()[:1]:  # Just show first sequence
                seq_mask = keys[:, 0] == seq_id
                seq_entries = keys[seq_mask]
                seq_tokens = tokens[seq_mask]
                
                print(f"  Seq {seq_id.item()}: {seq_mask.sum().item()} entries", flush=True)
                
                # Show first 2 unique recovery tokens
                for rec_token in seq_entries[:, 2].unique()[:2]:
                    rec_mask = seq_entries[:, 2] == rec_token
                    if rec_mask.any():
                        idx = rec_mask.nonzero(as_tuple=True)[0][0]
                        k_idx = seq_entries[idx, 1].item()
                        
                        rec_text = self.tokenizer.decode([rec_token.item()])
                        spec_tokens = seq_tokens[idx].tolist()
                        spec_text = [self.tokenizer.decode([t]) for t in spec_tokens]
                        print(f"    k={k_idx}, rec={rec_token.item()} ('{rec_text}') -> {spec_text}", flush=True)
            print(f"{'='*80}\n", flush=True)
    
    def _start_interrupt_listener(self):  # TODO: do we need an irecv here? do we use it? or should we just listen after we finish tree decoding
        """Initiates a non-blocking receive for the next command to allow interruption."""
        cmd_tensor = torch.empty(1, dtype=torch.int64, device=self.device)
        work_handle = dist.irecv(cmd_tensor, src=0, group=self.async_pg)
        # return both the handle and its tensor buffer
        return work_handle, cmd_tensor

    # new one, with true asynchrony
    def draft_loop(self):
        """
        Runs the asynchronous draft model loop. 
        Handles three commands:
          1 = prefill, 0 = spec request, 2 = exit, 3 = branch prefetch (only after a spec request).
        """
        assert self.draft_async, "draft_loop only runs in async-draft mode"

        while True:
            # 1) Wait for the next command (may be PREFILL, SPEC_REQUEST, or EXIT)
            cmd = self.recv_cmd()

            # PREFILL: run the draft prefill and then loop back
            if cmd == 1:
                self.draft_async_prefill()
                continue

            # SPECULATE request: serve out-of-cache or random speculations
            elif cmd == 0:
                _ds0 = time.perf_counter()
                _prof = os.environ.get("SSD_PROFILE", "0") == "1"
                if _prof:
                    torch.cuda.synchronize()
                    _d0 = time.perf_counter()

                glue_decode_input_ids, partial_tree_decode_args = self._service_spec_request()  # ~3ms

                if _prof:
                    torch.cuda.synchronize()
                    _d1 = time.perf_counter()

                self._reset_tree_cache_tensors()

                tree_decode_args = self._build_tree_batch(partial_tree_decode_args, glue_decode_input_ids)  # ~3ms --> leaves ~28ms for decoding

                if _prof:
                    torch.cuda.synchronize()
                    _d2 = time.perf_counter()

                # Decode the branch tree
                tokens, logits, activations = self._decode_tree(tree_decode_args)

                if _prof:
                    torch.cuda.synchronize()
                    _d3 = time.perf_counter()

                # Populate the local cache so future spec-requests can hit
                self._populate_tree_cache(tree_decode_args, tokens, logits, tree_decode_args["cache_hits"], activations)
                self._draft_step_times.append(time.perf_counter() - _ds0)

                if _prof:
                    torch.cuda.synchronize()
                    _d4 = time.perf_counter()
                    print(f"[PROFILE draft] service={(_d1-_d0)*1000:.2f}ms build_tree={(_d2-_d1)*1000:.2f}ms decode_tree={(_d3-_d2)*1000:.2f}ms populate={(_d4-_d3)*1000:.2f}ms total={(_d4-_d0)*1000:.2f}ms", flush=True)

                continue

            # EXIT: clean up and break out of the loop
            elif cmd == 2:
                if self._draft_step_times:
                    avg_ms = sum(self._draft_step_times) * 1000 / len(self._draft_step_times)
                    print(f"[metrics] Avg draft step time (ms): {avg_ms:.2f}", flush=True)
                self.exit()
                break

            else:
                raise RuntimeError(f"draft_loop: unknown command {cmd}")
