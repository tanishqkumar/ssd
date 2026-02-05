from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from ssd.engine.model_runner import ModelRunner
from ssd.engine.sequence import Sequence
from ssd.engine.scheduler import Scheduler
from ssd.engine.helpers.speculate_types import SpeculatorBase, VerifierBase, VerifyResult


class InferenceStep(ABC):

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    @abstractmethod
    def decode(self, seqs: list[Sequence]) -> int:
        pass

    @abstractmethod
    def prefill(self, seqs: list[Sequence]) -> int:
        pass


class AutoRegressiveStep(InferenceStep):

    def __init__(self, scheduler: Scheduler, model_runner: ModelRunner):
        super().__init__(scheduler)
        self.model_runner = model_runner

    def step(self, seqs: list[Sequence], is_prefill: bool) -> int:
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return len(seqs) if not is_prefill else sum(len(seq) for seq in seqs)

    def prefill(self, seqs: list[Sequence]) -> int:
        return self.step(seqs, is_prefill=True)

    def decode(self, seqs: list[Sequence]) -> int:
        return self.step(seqs, is_prefill=False)


class SpecDecodeStep(InferenceStep):

    def __init__(self, scheduler: Scheduler, speculator: SpeculatorBase, verifier: VerifierBase, eagle: bool):
        super().__init__(scheduler)
        self.speculator = speculator
        self.verifier = verifier
        self.eagle = eagle

    def prefill(self, seqs: list[Sequence]) -> int:
        verify_result = self.verifier.prefill(seqs, eagle=self.eagle)
        speculate_result = self.speculator.prefill(verify_result, eagle=self.eagle)
        return sum(len(seq) for seq in speculate_result.seqs_copy)

    def decode(self, seqs: list[Sequence]) -> int:
        in_verify_result = VerifyResult(
            seqs_copy=seqs,
            new_suffixes=[],
            recovery_tokens=[],
            eagle_acts=None,
        )
        speculate_result = self.speculator.speculate(in_verify_result, eagle=self.eagle)
        out_verify_result = self.verifier.verify(speculate_result, eagle=self.eagle)
        # handles BOTH block managers and seq state

        # TODO: Think about this abstraction
        self.scheduler.postprocess_speculate(
            out_verify_result.seqs_copy,
            out_verify_result.new_suffixes,
            out_verify_result.recovery_tokens,
        )
        return sum(len(s) for s in out_verify_result.new_suffixes)
