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
        )
        return draft_cfg

    def __init__(self, cfg: Config, rank: int = 0, init_q = None):
        self.draft_cfg = self.create_draft_config(cfg)
        self.is_draft = True # this is is_draft, use self.config.draft for the draft model path 
        self.prev_num_tokens = None
        super().__init__(self.draft_cfg, rank=rank, event=None, is_draft=True, num_tp_gpus=1, init_q=init_q)

        if self.is_draft and self.draft_async:
            self._reset_tree_cache_tensors()
            print(f'DraftRunner set up, starting draft_loop', flush=True)
            self.draft_loop()

    
    def draft_async_prefill(self):
        assert self.draft_async and self.is_draft

        # 1) Receive metadata
        metadata = torch.zeros(5, dtype=torch.int64, device=self.device)
        dist.recv(metadata, src=0, group=self.async_pg)
        total_new_tokens, total_slots, max_q, max_k, batch_size = metadata.tolist()

        # 2) allocate zeros tensors of the exact same shape and device
        input_ids = torch.zeros(
            total_new_tokens, dtype=torch.int64, device=self.device)
        positions = torch.zeros(
            total_new_tokens, dtype=torch.int64, device=self.device)
        cu_q = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=self.device)
        cu_k = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=self.device)
        slot_map = torch.zeros(
            total_slots, dtype=torch.int32, device=self.device)
        max_blocks = (self.config.max_model_len +
                      self.block_size - 1) // self.block_size
        block_tables = torch.zeros( 
            batch_size, max_blocks, dtype=torch.int32, device=self.device)

        # 3) receive in the exact same order:
        for t in (input_ids, positions, cu_q, cu_k, slot_map, block_tables):
            dist.recv(t, src=0, group=self.async_pg)

        # 4) receive eagle_acts if use_eagle
        eagle_acts = None
        if self.config.use_eagle:
            eagle_acts = torch.zeros(total_new_tokens, 3 * self.config.d_model_target, 
                                     dtype=torch.float16, device=self.device)
            dist.recv(eagle_acts, src=0, group=self.async_pg)
            assert eagle_acts is not None, "Draft must receive eagle_acts when use_eagle is True"
            print(f'[draft_async_prefill] METADATA: total_new_tokens={total_new_tokens}, total_slots={total_slots}, max_q={max_q}, max_k={max_k}, batch_size={batch_size}', flush=True)
            print(f'[draft_async_prefill] eagle_acts.shape={eagle_acts.shape}, input_ids.shape={input_ids.shape}', flush=True)
            
            # Map target vocab tokens to draft vocab using t2d
            assert hasattr(self.model, 't2d_tensor') and self.model.t2d_tensor is not None, "t2d_tensor not loaded"
            input_ids = self.model.t2d_tensor[input_ids]
            print(f'[draft_async_prefill] Mapped input_ids from target vocab to draft vocab', flush=True)

        # 5) set up context exactly like prepare_prefill() does:
        set_context(is_prefill=True, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_q, max_seqlen_k=max_k,
                    slot_mapping=slot_map, context_lens=None) # , block_tables=block_tables, commenting this out essentially removes prefix caching

        # 6) run the draft model in prefill mode
        self.run_model(input_ids, positions, is_prefill=True, last_only=True, hidden_states=eagle_acts) 
        print(f'[draft_async_prefill] DRAFT PREFILL DONE', flush=True)

        # 7) clean up
        reset_context()

    def _reset_tree_cache_tensors(self):
        """Reset tensor-backed tree cache to empty."""
        # initialize as empty keys on correct device; tokens/logits set to None until first populate
        self.tree_cache_keys = torch.zeros( 
            (0, 3), dtype=torch.int64, device=self.device)
        self.tree_cache_tokens = None
        self.tree_cache_logits = None

    def jit_speculate(self, 
                      request_keys: torch.Tensor, 
                      num_tokens: torch.Tensor, 
                      out_logits: torch.Tensor, 
                      out_tokens: torch.Tensor, 
                      temperatures: torch.Tensor, 
                      draft_block_tables: torch.Tensor): # should run K+1 steps starting at nt + rq[i, 1]
        
        # input_ids = rq[i, -1] the new rec tokens 
        input_ids = request_keys[:, -1] # [B]
        positions = num_tokens - 1 # [B]
        context_lens = positions + 1 # [B]
        # Calculate slot mapping vectorized
        block_idx = positions // self.block_size
        pos_in_block = positions % self.block_size
        batch_indices = torch.arange(input_ids.shape[0], device=self.device)
        slot_map = draft_block_tables[batch_indices, block_idx] * self.block_size + pos_in_block

        for i in range(self.config.speculate_k): # we're going to glue after this anyways 
            set_context(
                is_prefill=False,
                slot_mapping=slot_map,
                context_lens=context_lens.to(torch.int32),
                block_tables=draft_block_tables,
                is_jit=True,
            )
            out_logits[:, i, :] = self.run_model(input_ids, positions, is_prefill=False, last_only=True)
            reset_context()
            next_tokens = self.sampler(out_logits[:, i, :], temperatures, is_tree=True)
            out_tokens[:, i] = next_tokens
            
            # Update for next iteration
            input_ids = next_tokens
            positions = positions + 1
            context_lens = context_lens + 1
            # Update slot mapping for next position
            block_idx = positions // self.block_size
            pos_in_block = positions % self.block_size
            slot_map = draft_block_tables[batch_indices, block_idx] * self.block_size + pos_in_block

        return

    def hit_cache_and_respond(self, request_keys, B, K, num_tokens, temperatures, draft_block_tables):
        """Hits the cache (tensor-backed) and returns tensors to respond to the spec request."""
        global ttl, ttl_hit
        V = self.hf_config.vocab_size

        torch.manual_seed(0)
        out_logits = torch.randn( 
            (B, K, V), dtype=self.hf_config.torch_dtype, device=self.device)
        out_tokens = torch.argmax(out_logits, dim=-1)[:, :K] # [B, K]
        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)

        assert request_keys.shape == (B, 3), f"ERROR in hit_cache_and_respond: request_keys should be (B, 3), got {request_keys.shape}"
        
        # Statistics
        ttl += int(B)
        
        if self.tree_cache_keys.numel() > 0:
            # Vectorized membership against tensor cache
            eq = (request_keys.unsqueeze(1) == self.tree_cache_keys.unsqueeze(0))  # [B,T,3]
            match = torch.all(eq, dim=2)  # [B,T]
            cache_hits = match.any(dim=1)  # [B]
            ttl_hit += int(cache_hits.sum().item())
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
            elif self.config.jit_speculate: 
                # print(f'[hit_cache_and_respond] found a cache miss, running jit speculate', flush=True)
                self.jit_speculate(
                    request_keys, 
                    num_tokens, 
                    out_logits, 
                    out_tokens,
                    temperatures,
                    draft_block_tables
                    ) # write into out_logits, out_tokens
            
        rec_toks = request_keys[:, 2]
        
        return out_tokens, out_logits, make_glue_decode_input_ids(out_tokens, rec_toks), cache_hits # gives [B*(K+1)] glue decode tokens

    def _service_spec_request(self):

        """Receives a speculation request, serves it from cache, and sends results back in a single response."""
        meta = self.recv_tensor((3,), torch.int64)
        # not going to get any preempted or finished seqs here since we just postprocessed+scheduled before this
        B, K, F = meta.tolist()

        # Receive all request payload in one fused int64 burst then temperatures
        max_blocks = (self.config.max_model_len +
                      self.block_size - 1) // self.block_size
        fused_total = (3 * B) + B + (B * max_blocks)
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
        assert off == fused_total
        temperatures = self.recv_tensor((B,), torch.float32)

        out_tokens, out_logits, glue_decode_input_ids, cache_hits = self.hit_cache_and_respond(
            cache_keys, 
            B, 
            K, 
            num_tokens, 
            temperatures, 
            draft_block_tables
            )

        # Send response: cache_hits and spec_tokens_flat in one fused message
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
        } 

        # from now on, need to thread cache_hits through tree decode and not assume that MQ_LEN is global constant, but varies with iteration depending on prev hit/miss
        return glue_decode_input_ids, partial_tree_decode_args 

    
    def prepare_glue_decode_ctxt(self, num_tokens, input_ids, dbt, B):
        """Prepares the VERIFY context for glue forward (varlen decode).

        num_tokens = num tokens in sequence, incl newly appended rec token not yet forwarded through draft 
        input_ids = for glue decode, shape [B * (K + 1)] since we pass in flat 
        dbt = draft block tables, shape [B, M]
        B = batch size

        Builds varlen decode tensors so that for each sequence i (out of B):
        - query length = K + 1
        - key length   = num_tokens[i] - 1 + (K + 1)
        We also compute slot_mapping for exactly the K+1 query tokens, so they
        are written into the KV cache in their correct slots.
        """

        assert num_tokens.shape == (
            B,), f"ERROR in prepare_glue_decode_ctxt: num_tokens should be (B,), got {num_tokens.shape}"

        K = self.config.speculate_k
        
        # Grid of absolute positions for K+1 tokens
        positions_start = (num_tokens - 1).unsqueeze(-1)  # [B, 1]
        positions_grid = positions_start + torch.arange(K + 1, device=self.device)  # [B, K+1]

        # Calculate block indices and offsets for ALL positions
        block_indices = (positions_grid //
                         self.block_size).to(torch.int64)  # [B, K+1]
        offsets = (positions_grid % self.block_size).to(
            torch.int32)  # [B, K+1]

        # Get block IDs for each position from dbt
        B_expanded = torch.arange(B, device=self.device).unsqueeze(
            -1).expand(-1, K + 1)  # [B, K+1]
        blk_ids = dbt[B_expanded, block_indices]  # [B, K+1]

        assert blk_ids.shape == (B, K + 1), f"ERROR in prepare_glue_decode_ctxt: blk_ids should be (B, {K + 1}), got {blk_ids.shape}"
        assert (blk_ids >= 0).all(
        ), "ERROR in prepare_glue_decode_ctxt: all blk_ids should be >= 0"

        # Calculate slot_map for each position
        slot_map_grid = blk_ids * self.block_size + offsets  # [B, K+1]

        # Flattened tensors for varlen decode
        positions_flat = positions_grid.reshape(-1).to(torch.int64)  # [B * (K+1)]
        slot_map_flat = slot_map_grid.reshape(-1).to(torch.int32)  # [B * (K+1)]

        assert input_ids.dim() == 1 == positions_flat.dim() == slot_map_flat.dim(), \
            f"input_ids, positions_flat, and slot_map_flat should all be 1D, got shapes {input_ids.shape}, {positions_flat.shape}, {slot_map_flat.shape}"
        
        # Build cu_seqlens_q
        seqlen_q = torch.full((B,), K + 1, dtype=torch.int32, device=self.device)  # [B]
        context_lens = ((num_tokens - 1) + (K + 1)).to(torch.int32)  # [B]
        cu_seqlens_q = torch.zeros(
            B + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)

        # print(f'[prepare_glue_decode_ctxt] input_ids: {[self.tokenizer.decode([token_id]) for token_id in input_ids.tolist()]}', flush=True)
        glue_decode_context = {
            "input_ids": input_ids,              # [B * (K+1)]
            "positions": positions_flat,              # [B * (K+1)]
            "slot_map": slot_map_flat,                # [B * (K+1)]
            "cu_seqlens_q": cu_seqlens_q,            # [B+1]
            "max_seqlen_q": K + 1,                   # [1]
            "context_lens": context_lens,            # [B], 
            "block_tables": dbt,                     # [B, M], these have long megaspec space allocated for each branch 
        }

        return glue_decode_context

    def _construct_tree_decode_args(self, partial_tree_decode_args, rec_flat, dbt):
        # tree decode needs (input_ids, positions) that are [N], wrapper plan handles batch size of attn computation 
        # rec_flat is [N]
        
        B = dbt.shape[0]
        K = self.config.speculate_k
        F = self.config.async_fan_out
        N = rec_flat.shape[0]
        cache_hits = partial_tree_decode_args["cache_hits"]

        assert N == B*self.config.MQ_LEN, f"ERROR in _construct_tree_decode_args: N should be B*self.config.MQ_LEN={B*self.config.MQ_LEN}, got {N}"
        assert self.config.fan_out_t.sum() == self.config.fan_out_t_miss.sum() == self.config.MQ_LEN, f"ERROR in _construct_tree_decode_args: fan_out_t.sum() should be MQ_LEN={self.config.MQ_LEN}, got {self.config.fan_out_t.sum()}"
        # assert N == B * (K + 1) * F, "ERROR in _construct_tree_decode_args: N should be B * (K + 1) * F"
        # b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[:, None, None].expand(B, K + 1, F).flatten()
        
        b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[:, None].expand(B, self.config.MQ_LEN).flatten()
        assert b_flat.shape == (N,), f"ERROR in _construct_tree_decode_args: b_flat should be (N,), got {b_flat.shape}"
        # fkp1_flat = torch.arange(F * (K + 1), device=self.device, dtype=torch.int64).repeat(B) 
        fkp1_flat = torch.arange(self.config.MQ_LEN, device=self.device, dtype=torch.int64).repeat(B) 
        assert fkp1_flat.shape == (N,), f"ERROR in _construct_tree_decode_args: fkp1_flat should be (N,), got {fkp1_flat.shape}"
        
        seq_ids = partial_tree_decode_args["seq_ids"] # [B]
        seq_ids_expanded = seq_ids[b_flat]
        assert seq_ids_expanded.shape == (N,), f"ERROR in _construct_tree_decode_args: seq_ids_expanded should be (N,), got {seq_ids_expanded.shape}"
        positions = (partial_tree_decode_args["num_tokens"][b_flat] - 1) + (K + 1) + fkp1_flat # this is crucial to get right, differs from sq/batch fan out
        # make rope_positions which all start at after K+1 of the glue decode are done 
        # j_idx_flat = fkp1_flat // F # this assumes unif fan out over lookahead
        # j_idx_flat = torch.arange(
        #     K+1, device=self.device, dtype=torch.int64).repeat_interleave(self.config.fan_out_t).repeat(B)  # i think this is right? 
        j_idx_flat = torch.cat([
            torch.arange(K+1, device=self.device, dtype=torch.int64).repeat_interleave(
                self.config.fan_out_t if hit else self.config.fan_out_t_miss)
            for hit in cache_hits
        ])
        assert j_idx_flat.shape == (N,), f"ERROR in _construct_tree_decode_args: j_idx_flat should be (N,), got {j_idx_flat.shape}"
        rope_positions = (partial_tree_decode_args["num_tokens"][b_flat] - 1) + j_idx_flat + 1
        assert rope_positions.shape == (N,), f"ERROR in _construct_tree_decode_args: rope_positions should be (N,), got {rope_positions.shape}"
        temperatures = partial_tree_decode_args["temperatures"][b_flat]

        # metadata needs B K F N
        metadata = torch.tensor([B, K, F, N], dtype=torch.int64, device=self.device)

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
        B = glue_decode_input_ids.shape[0] // (K + 1)

        assert B == partial_tree_decode_args["num_tokens"].shape[
            0], "ERROR in _build_tree_batch: B should be the same as the number of tokens"

        # Prepare context for glue decode
        dbt = partial_tree_decode_args["dbt"]
        
        glue_decode_ctxt = self.prepare_glue_decode_ctxt(
            num_tokens=partial_tree_decode_args["num_tokens"],
            input_ids=glue_decode_input_ids,
            dbt=dbt,
            B=B, 
        ) 

        set_context(
            is_prefill=False,
            cu_seqlens_q=glue_decode_ctxt["cu_seqlens_q"],
            max_seqlen_q=glue_decode_ctxt["max_seqlen_q"], 
            slot_mapping=glue_decode_ctxt["slot_map"],
            context_lens=glue_decode_ctxt["context_lens"],
            block_tables=glue_decode_ctxt["block_tables"]
        )

        glue_decode_logits_flat = self.run_model(
            glue_decode_ctxt["input_ids"], glue_decode_ctxt["positions"], is_prefill=False, last_only=False)  # runs the verify cudagraph
        reset_context()
    
        glue_decode_logits = glue_decode_logits_flat.view(B, K+1, -1) # [B, K+1, V]
        forked_rec_tokens = get_forked_recovery_tokens_from_logits(
            self.config,
            glue_decode_logits,
            partial_tree_decode_args["cache_hits"],
            glue_decode_input_ids.reshape(B, K+1), # [B, K+1], we set probs of these to 0 when forking to avoid forking them 
            tokenizer=self.tokenizer
        ).view(-1)  # [B*F*(K+1)] = [N], now [B*MQ_LEN]

        return self._construct_tree_decode_args(partial_tree_decode_args, forked_rec_tokens, dbt)

    @torch.inference_mode()
    def _compute_step_positions_and_slot_maps(self, initial_positions, initial_rope_positions, dbt, B, K, F, N, MQ_LEN):
        """Precompute positions and slot maps for all K steps - batch size independent logic."""
        # Precompute positions for all steps: [K, N]
        step_positions = initial_positions[None, :] + torch.arange(K, device=self.device)[:, None] * MQ_LEN
        step_rope_positions = initial_rope_positions[None, :] + torch.arange(K, device=self.device)[:, None]
        
        # Precompute context_lens for all steps: [K, B]
        step_context_lens = step_positions.view(K, B, MQ_LEN)[:, :, -1] + 1  # [K, B]
        
        # Precompute slot_maps for all steps: [K, N]
        # b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[:, None, None].expand(B, K+1, F).flatten()  # [N], be careful here!
        b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[
            :, None].expand(B, self.config.MQ_LEN).flatten()
        dbt_expanded = dbt[b_flat]  # [N, M] - constant across steps
        
        step_offsets = (step_positions % self.block_size).to(torch.int32)  # [K, N]
        step_last_blks = (step_positions // self.block_size).to(torch.int64)  # [K, N]
        batch_indices = torch.arange(N, device=self.device)  # [N]
        step_blk_ids = dbt_expanded[batch_indices[None, :], step_last_blks]  # [K, N]
        step_slot_maps = step_blk_ids * self.block_size + step_offsets  # [K, N]
        
        return step_positions, step_rope_positions, step_context_lens, step_slot_maps

    def _decode_tree_step(self, depth, current_input_ids, step_rope_positions, step_slot_maps, step_context_lens, dbt, payload, spec_tokens, spec_logits):
        """Execute a single tree decode step."""
        # Use precomputed values for this step
        set_context(
            is_prefill=False,
            slot_mapping=step_slot_maps[depth],
            context_lens=step_context_lens[depth].to(torch.int32),
            block_tables=dbt,
        )

        logits = self.run_model(current_input_ids, step_rope_positions[depth], is_prefill=False, last_only=False, tree_decode_step=depth, cache_hits=payload["cache_hits"])
        reset_context()
        
        logits_flat = logits.view(-1, self.hf_config.vocab_size)  # [N, V]
        spec_logits[:, depth, :] = logits_flat
        next_tokens = self.sampler(logits_flat, payload["temps"], is_tree=True)
        spec_tokens[:, depth] = next_tokens
        
        return next_tokens

    def _decode_tree_interruptibly(self, payload):
        """Decodes the speculation tree, checking for interrupts at each step."""

        # setup
        metadata = payload["metadata"]
        B, K, F, N = metadata[0].item(), metadata[1].item(
        ), metadata[2].item(), metadata[3].item()

        V = self.hf_config.vocab_size
        spec_tokens = torch.empty(
            (N, K), dtype=torch.int64, device=self.device)
        spec_logits = torch.empty(
            (N, K, V), dtype=self.hf_config.torch_dtype, device=self.device)

        # Precompute all positions, context_lens, and slot_maps for all K steps
        initial_positions = payload["positions"].clone()  # [N]
        initial_rope_positions = payload["rope_positions"].clone() # [N]
        current_input_ids = payload["input_ids"]  # [N], the forked tokens
        dbt = payload["block_tables"]  # [B, M] - constant across steps
        
        # Use compiled function for batch-size independent computations
        _, step_rope_positions, step_context_lens, step_slot_maps = self._compute_step_positions_and_slot_maps(
            initial_positions, initial_rope_positions, dbt, B, K, F, N, self.config.MQ_LEN
        )

        for depth in range(K):
            current_input_ids = self._decode_tree_step(
                depth, current_input_ids, step_rope_positions, step_slot_maps, 
                step_context_lens, dbt, payload, spec_tokens, spec_logits
            )

        return spec_tokens, spec_logits

    def _populate_tree_cache(self, payload, tokens, logits, cache_hits):
        """Populates the tensor-backed tree_cache with the results of the decoding.
        """
        seq_ids_expanded = payload["seq_ids_expanded"].to(torch.int64)
        rec_flat = payload["rec_flat"].to(torch.int64)

        # B = payload["block_tables"].shape[0]
        # k_flat = torch.arange(self.config.speculate_k + 1, device=self.device, dtype=torch.int64)[None, :, None].expand(
        #     B, self.config.speculate_k + 1, self.config.async_fan_out).flatten()

        # TODO: make sure the cache req we send from target is aware of last cache_hits?
        k_flat_hit = torch.arange(self.config.speculate_k+1, device=self.device, dtype=torch.int64).repeat_interleave(
            self.config.fan_out_t)
        k_flat_miss = torch.arange(self.config.speculate_k+1, device=self.device, dtype=torch.int64).repeat_interleave(
            self.config.fan_out_t_miss)
        k_flat = torch.cat([
            k_flat_hit if hit else k_flat_miss
            for hit in cache_hits
        ])

        assert k_flat.shape[0] == payload["block_tables"].shape[0] * self.config.MQ_LEN, f"ERROR in _populate_tree_cache: k_flat should be {payload['block_tables'].shape[0] * self.config.MQ_LEN}, got {k_flat.shape[0]}"
        
        keys = torch.stack([seq_ids_expanded, k_flat, rec_flat], dim=1).contiguous()  # [N,3]

        assert self.tree_cache_keys.numel() == 0
        self.tree_cache_keys = keys
        self.tree_cache_tokens = tokens
        self.tree_cache_logits = logits
    
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
                print(f'[draft] PREFILL command received on draft', flush=True)
                self.draft_async_prefill()
                continue

            # SPECULATE request: serve out-of-cache or random speculations
            elif cmd == 0:
                glue_decode_input_ids, partial_tree_decode_args = self._service_spec_request()  # ~3ms

                self._reset_tree_cache_tensors()

                tree_decode_args = self._build_tree_batch(partial_tree_decode_args, glue_decode_input_ids)  # ~3ms --> leaves ~28ms for decoding

                # Decode the branch tree (with early interruption)
                tokens, logits = self._decode_tree_interruptibly(tree_decode_args)

                # Populate the local cache so future spec-requests can hit
                self._populate_tree_cache(tree_decode_args, tokens, logits, tree_decode_args["cache_hits"])

                continue

            # EXIT: clean up and break out of the loop
            elif cmd == 2:
                print(f'[draft] EXIT command received', flush=True)
                self.exit()
                break

            else:
                raise RuntimeError(f"draft_loop: unknown command {cmd}")

