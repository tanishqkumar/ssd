from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_new_tokens: int = 256
    ignore_eos: bool = False
    # Async-spec only: allow distinct temps for draft (tree decode) and target (verify)
    # If None, they default to `temperature`.
    draft_async_temperature: float | None = None
    target_async_temperature: float | None = None
