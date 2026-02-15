import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase
from ssd.engine.helpers.runner_helpers import prepare_prefill_tensors_from_seqs
from ssd.engine.helpers.handshake_helpers import TargetDraftHandshake
from ssd.engine.sequence import Sequence
from ssd.utils.misc import decode_tokens


class SpeculatorAsync(SpeculatorBase):

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        async_fan_out: int,
        max_blocks: int,
        vocab_size: int,
        draft_dtype: torch.dtype,
        kvcache_block_size: int,
        max_model_len: int,
        async_pg: dist.ProcessGroup,
        draft_runner_rank: int,
        tokenizer: AutoTokenizer,
        verbose: bool,
    ):
        super().__init__(lookahead, device)
        self.async_fan_out = async_fan_out
        self.max_blocks = max_blocks
        self.vocab_size = vocab_size
        self.draft_dtype = draft_dtype
        self.kvcache_block_size = kvcache_block_size
        self.max_model_len = max_model_len
        self.async_pg = async_pg
        self.draft_runner_rank = draft_runner_rank
        self.tokenizer = tokenizer
        self.verbose = verbose

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        eagle_acts = verify_result.eagle_acts
        skip_first = 1 if eagle_acts is not None else 0

        # 1) build all the prefill payload in one shot
        input_ids, positions, cu_q, cu_k, max_q, max_k, slot_map = \
            prepare_prefill_tensors_from_seqs(
                seqs,
                block_size=self.kvcache_block_size,
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
        max_blocks = (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size
        block_tables = torch.tensor(
            [s.draft_block_table + [-1] *
                (max_blocks - len(s.draft_block_table)) for s in seqs],
            dtype=torch.int32, device=self.device,
        )
        
        # 3) send cmd=1
        cmd = torch.tensor([1], dtype=torch.int64, device=self.device)
        dist.send(cmd, dst=self.draft_runner_rank, group=self.async_pg)

        # 4) send metadata for tensor reconstruction
        metadata = torch.tensor([
            input_ids.size(0),
            slot_map.size(0),
            max_q,
            max_k,
            len(seqs),  # batch_size
        ], dtype=torch.int64, device=self.device)
        dist.send(metadata, dst=self.draft_runner_rank, group=self.async_pg)

        # 5) send each tensor in a fixed order
        for t in (input_ids, positions, cu_q, cu_k, slot_map, block_tables):
            dist.send(t, dst=self.draft_runner_rank, group=self.async_pg)
        
        # 6) send eagle_acts if use_eagle (cast to draft dtype to match receive buffer)
        if eagle_acts is not None:
            dist.send(eagle_acts.to(self.draft_dtype), dst=self.draft_runner_rank, group=self.async_pg)

        return SpeculateResult([], [])

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        """
        - Hit the cache with cache_keys, get speculations and logits
            - This will involve deallocating past failed fork blocks, and allocating new forks, and coordinating this with the draft runner
                - But this is abstracted away at this level 
        - Send info to help DraftRunner build tree cache (everything make_branch_bt needs)
            - This includes using helpers that eg. allocate blocks for the forked sequences 
            - And sending context we built to DraftRunner via nccl 
        """
        # Append recovery tokens to local seqs to correctly calculate initial positions
        for seq in seqs:
            if seq.recovery_token_id is None:
                raise ValueError(
                    f"recovery_token_id is None for seq {seq.seq_id}")
            seq.append_token(seq.recovery_token_id)

        # Log sequence trunk state for debugging
        if self.verbose:
            print(f"\n{'='*80}", flush=True)
            print(f"[TARGET SEQUENCE TRUNK] Batch size: {len(seqs)}", flush=True)
            for i, seq in enumerate(seqs):
                # Show last 20 tokens of trunk + recovery token
                trunk_tokens = seq.token_ids[-20:] if len(seq.token_ids) > 20 else seq.token_ids
                trunk_text = decode_tokens(trunk_tokens, self.tokenizer)
                recovery_text = decode_tokens([seq.recovery_token_id], self.tokenizer)
                print(f"  Seq {seq.seq_id} (len={len(seq.token_ids)}):", flush=True)
                print(f"    Trunk (last 20): ...{trunk_text}", flush=True)
                print(f"    Recovery token: {seq.recovery_token_id} ('{recovery_text}')", flush=True)
            print(f"{'='*80}\n", flush=True)


        # hit draft tree cache via handshake API, this bounds them
        eagle = verify_result.eagle_acts is not None
        speculations_tokens, logits_q, cache_hits = self.target_draft_handshake(seqs, eagle)

        # The first column of the final speculation tensor is the recovery tokens
        recovery_tokens_tensor = torch.tensor(
            [seq.recovery_token_id for seq in seqs], dtype=torch.int64, device=self.device)
        # Assert that no recovery tokens are None
        assert not torch.any(recovery_tokens_tensor == -
                             1), f"Found None recovery tokens in tensor: {recovery_tokens_tensor}"
        assert recovery_tokens_tensor.numel() > 0, "Recovery tokens tensor is empty"
        speculations = torch.cat(
            [recovery_tokens_tensor.unsqueeze(1), speculations_tokens], dim=1)

        # Update seqs with all speculated tokens for the verify step to pass through the target model
        for i, seq in enumerate(seqs):
            # we added rec token first thing
            seq.token_ids.extend(speculations_tokens[i].tolist())
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = seq.token_ids[-1]
            assert len(speculations_tokens[i]) == self.lookahead, (
                f"speculations_tokens[i] should have length {self.lookahead}, "
                f"got {len(speculations_tokens[i])}"
            )
            seq.num_draft_cached_tokens += len(speculations_tokens[i]) + 1

        # speculations is [B, K+1] at this point since we prepending seq.recovery_token_id for
        return SpeculateResult(speculations, logits_q, cache_hits)


    def target_draft_handshake(self, seqs: list[Sequence], eagle: bool):
        """Send a spec request command with cache keys to get speculations/logits."""

        # Use the handshake helper to handle the complete protocol
        handshake = TargetDraftHandshake(
            seqs=seqs,
            lookahead=self.lookahead,
            async_fan_out=self.async_fan_out,
            max_blocks=self.max_blocks,
            eagle=eagle,
            vocab_size=self.vocab_size,
            draft_dtype=self.draft_dtype,
            device=self.device,
            async_pg=self.async_pg,
            draft_runner_rank=self.draft_runner_rank
        )
        speculations, logits_q, cache_hits = handshake.execute_full_handshake()

        return speculations, logits_q, cache_hits
