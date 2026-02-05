import torch
from ssd.engine.sequence import Sequence
from ssd.engine.model_runner import ModelRunner
from ssd.utils.verify import verify
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, VerifierBase


class Verifier(VerifierBase):
    def __init__(self, lookahead: int, device: torch.device, target_model_runner: ModelRunner):
        super().__init__(lookahead, device)
        self.target_model_runner = target_model_runner

    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        # TODO: Eagle
        result = self.target_model_runner.call("run", seqs, True)
        if eagle:
            token_ids, eagle_acts = result
        else:
            token_ids = result

        offset = 0
        for seq, token_id in zip(seqs, token_ids):
            seq.recovery_token_id = token_id
            if eagle:
                seq_len = seq.num_prompt_tokens
                # this doesn't move acts onto cpu does it? 
                seq.last_target_hidden_state = eagle_acts[offset + seq_len - 1].clone()
                offset += seq_len

        return VerifyResult(
            seqs,
            [], # no accepted tokens for prefill, just recovery tokens (first sampled token).
            [seq.recovery_token_id for seq in seqs],
            eagle_acts if eagle else None,
        )

    def verify(self, speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        """Verify speculative tokens using the target model."""
        seqs_copy = speculate_result.seqs_copy
        batch_size = len(seqs_copy)

        result = self.target_model_runner.call("run", seqs_copy, False, False, True)
        
        if eagle:
            logits_p_flat, eagle_acts_flat = result
        else:
            logits_p_flat = result

        for s in seqs_copy:  # was debuge, but is correct 
            s.num_cached_tokens += self.lookahead + 1

        logits_p = logits_p_flat.view(
            batch_size, self.lookahead + 1, -1)  # [b, k+1, v]

        # Build per-seq temps for target verify and draft q respectively.
        temps_target = temps_draft = [seq.temperature for seq in seqs_copy]
        temperatures_target = torch.tensor(temps_target, dtype=torch.float32, device=self.device)
        temperatures_draft = torch.tensor(temps_draft, dtype=torch.float32, device=self.device)

        new_suffixes, recovery_tokens = verify(
            logits_p,
            speculate_result.logits_q,
            speculate_result.speculations,
            temperatures_target,
            temperatures_draft,
        )

        # # Debug: print recovery tokens detokenized
        # if __debug__ and recovery_tokens is not None and len(recovery_tokens) > 0:
        #     tokenizer = self.tokenizer
        #     recovery_texts = []
        #     for token in recovery_tokens:
        #         try:
        #             text = tokenizer.decode([token], skip_special_tokens=False)
        #             recovery_texts.append(text)
        #         except Exception:
        #             recovery_texts.append(f"<token_id:{token}>")
        #     print(f"[verify] recovery tokens: {recovery_texts}", flush=True)
        
        # METRICS["accepted_suffix_lens_with_recovery"].extend(
        #     [len(s) for s in new_suffixes]) 

        # For async mode, also track accepted suffix lengths only for cache hits
        # if self.config.draft_async and cache_hits is not None:
        #     for i, suffix_len in enumerate([len(s) for s in new_suffixes]):
        #         if cache_hits[i] == 1:  # Cache hit
        #             METRICS["accepted_suffix_lens_on_hit"].append(suffix_len)
        #         else: # cache miss
        #             METRICS["accepted_suffix_lens_on_miss"].append(suffix_len)

        # Print mean length of new suffixes for monitoring
        if new_suffixes:
            mean_suffix_len = sum(len(suffix)
                                  for suffix in new_suffixes) / len(new_suffixes)
            if __debug__: print(f"[verify] mean new suffix length: {mean_suffix_len:.2f}", flush=True)

        eagle_acts = None
        if eagle:
            eagle_acts = eagle_acts_flat.view(batch_size, self.lookahead + 1, -1)
        
        return VerifyResult(
            seqs_copy=seqs_copy,
            new_suffixes=new_suffixes,
            recovery_tokens=recovery_tokens,
            eagle_acts=eagle_acts,
        )
