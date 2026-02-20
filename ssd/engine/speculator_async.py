import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase
from ssd.engine.helpers.runner_helpers import prepare_prefill_payload, send_speculation_request, receive_speculation_response
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
        self.K = lookahead

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        eagle_acts = verify_result.eagle_acts
        input_id_list = [seq.token_ids for seq in seqs]

        max_blocks = (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size
        cmd, metadata, input_ids, num_tokens, draft_block_table, eagle_acts = prepare_prefill_payload(
            input_id_list,
            eagle_acts,
            self.device,
            max_blocks,
            [seq.draft_block_table for seq in seqs],
        )

        dist.send(cmd, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(metadata, dst=self.draft_runner_rank, group=self.async_pg)
        for t in (input_ids, num_tokens, draft_block_table, eagle_acts):
            if t is not None:
                dist.send(t, dst=self.draft_runner_rank, group=self.async_pg)

        return SpeculateResult([], [])

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        for seq in seqs:
            if seq.recovery_token_id is None:
                raise ValueError(
                    f"recovery_token_id is None for seq {seq.seq_id}")
            seq.append_token(seq.recovery_token_id)

        if self.verbose:
            sep = '=' * 80; print(f"\n{sep}", flush=True)
            print(f"[TARGET SEQUENCE TRUNK] Batch size: {len(seqs)}", flush=True)
            for i, seq in enumerate(seqs):
                trunk_tokens = seq.token_ids[-20:] if len(seq.token_ids) > 20 else seq.token_ids
                trunk_text = decode_tokens(trunk_tokens, self.tokenizer)
                recovery_text = decode_tokens([seq.recovery_token_id], self.tokenizer)
                print(f"  Seq {seq.seq_id} (len={len(seq.token_ids)}):", flush=True)
                print(f"    Trunk (last 20): ...{trunk_text}", flush=True)
                print(f"    Recovery token: {seq.recovery_token_id} ({recovery_text})", flush=True)
            print(f"{sep}\n", flush=True)

        eagle = verify_result.eagle_acts is not None
        speculations_tokens, logits_q, cache_hits = self.speculation_request(seqs, eagle)

        recovery_tokens_tensor = torch.tensor(
            [seq.recovery_token_id for seq in seqs], dtype=torch.int64, device=self.device)
        speculations = torch.cat(
            [recovery_tokens_tensor.unsqueeze(1), speculations_tokens], dim=1)

        for i, seq in enumerate(seqs):
            seq.token_ids.extend(speculations_tokens[i].tolist())
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = seq.token_ids[-1]
            seq.num_draft_cached_tokens += len(speculations_tokens[i]) + 1

        return SpeculateResult(speculations, logits_q, cache_hits)

    def speculation_request(self, seqs: list[Sequence], eagle: bool):
        send_speculation_request(
            seqs=seqs,
            lookahead=self.lookahead,
            async_fan_out=self.async_fan_out,
            max_blocks=self.max_blocks,
            eagle=eagle,
            draft_dtype=self.draft_dtype,
            device=self.device,
            async_pg=self.async_pg,
            draft_runner_rank=self.draft_runner_rank,
        )

        speculations, logits_q, cache_hits = receive_speculation_response(
            batch_size=len(seqs),
            lookahead=self.lookahead,
            vocab_size=self.vocab_size,
            draft_dtype=self.draft_dtype,
            device=self.device,
            async_pg=self.async_pg,
            draft_runner_rank=self.draft_runner_rank,
        )

        return speculations, logits_q, cache_hits
