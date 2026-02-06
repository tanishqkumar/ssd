from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    draft_temperature: float | None = None
    max_new_tokens: int = 256
    ignore_eos: bool = False
