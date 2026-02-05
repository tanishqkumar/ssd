import torch
from ssd.engine.model_runner import ModelRunner
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase


class SpeculatorSync(SpeculatorBase):

    def __init__(self, lookahead: int, device: torch.device, draft_model_runner: ModelRunner):
        super().__init__(lookahead, device)
        self.draft_model_runner = draft_model_runner

    def prefill(self, verify_result: VerifyResult, eagle: bool = False) -> SpeculateResult:
        assert not eagle, "Eagle is not currently supported for synchronous speculation"
        self.draft_model_runner.call("run", verify_result.seqs_copy, True)
        # recovery token will be first token in next fwd, but not yet in kvc of either model
        for seq in verify_result.seqs_copy:
            assert seq.recovery_token_id is not None
            seq.num_cached_tokens = seq.num_prompt_tokens
            seq.num_draft_cached_tokens = seq.num_prompt_tokens

        return SpeculateResult(verify_result.seqs_copy, [], [])

    def speculate(self, verify_result: VerifyResult, eagle: bool = False) -> SpeculateResult:
        """Generate k speculative tokens using the draft model."""
        assert not eagle, "Eagle is not currently supported for synchronous speculation"

        # TODO: How do we get the target activations from the last verify?
        seqs_copy = [seq.clone_spec() for seq in verify_result.seqs_copy]

        batch_size = len(seqs_copy)

        speculations = torch.zeros(
            batch_size, self.lookahead + 1,
            dtype=torch.int64,
            device=self.device,
        )
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

        for k in range(self.lookahead + 1):
            # Draft model forward pass - emits [B] tokens, True is for draft_return_logits
            token_ids, step_logits_q = self.draft_model_runner.call(
                "run", seqs_copy, False, True, True)
            # make sure we include this even on last iter since we put K+1 tokens thru draft cache
            for s in seqs_copy:
                s.num_draft_cached_tokens += 1

            if k == self.lookahead:
                break  # this extra fwd also

            logits_q.append(step_logits_q)

            for i, (seq, token_id) in enumerate(zip(seqs_copy, token_ids)):
                seq.append_token(token_id)

            # Single batched write to GPU
            speculations[:, k + 1] = torch.tensor(
                token_ids, dtype=torch.int64, device="cuda")

        logits_q = torch.stack(logits_q, dim=1)  # [B, K, V]

        return SpeculateResult(seqs_copy, speculations, logits_q)
