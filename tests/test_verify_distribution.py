"""
Exactness test for ssd.utils.verify.verify(): the emitted token distribution must
equal the target distribution p.
"""
import os
import pathlib
import sys
import types

# Enable CPU-safe leaf module verify() import on a CPU-only venv.
os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
try:
    # Real package if GPU deps exist.
    import ssd  # noqa: F401
except Exception:
    # Skip ssd/__init__.py; load leaf modules only.
    _pkg = types.ModuleType("ssd")
    _pkg.__path__ = [str(_ROOT / "ssd")]
    sys.modules["ssd"] = _pkg

import torch
import pytest

from ssd.utils.verify import verify

V = 4
N = 200_000
P0 = torch.tensor([0.40, 0.30, 0.20, 0.10], dtype=torch.float64)
Q0 = torch.tensor([0.10, 0.40, 0.30, 0.20], dtype=torch.float64)


def _tv(a, b):
    """Total-variation distance between two distributions."""
    return 0.5 * (a - b).abs().sum().item()


def _onehot(logits):
    oh = torch.zeros_like(logits)
    oh[int(logits.argmax())] = 1.0
    return oh


def _law(logits, temp):
    return torch.softmax(logits / temp, dim=-1) if temp > 0 else _onehot(logits)


def _run(temp, cache_hits, jit, seed=0):
    """Draw N proposals x1 ~ q0 and run verify().
    
    Returns:
        A tuple (phat, p0), where phat is the empirical law of T and
        p0 is the target law.
    """
    torch.manual_seed(seed)
    p_logits, q_logits = P0.log(), Q0.log()
    p0, q0 = _law(p_logits, temp), _law(q_logits, temp)

    x1 = torch.multinomial(q0.expand(N, V), num_samples=1).squeeze(1)     # [N]
    speculations = torch.stack([torch.zeros(N, dtype=torch.int64), x1], dim=1)  # [N, K+1], K=1

    logits_p = p_logits.to(torch.float32).view(1, 1, V).expand(N, 2, V).contiguous()
    logits_q = q_logits.to(torch.float32).view(1, 1, V).expand(N, 1, V).contiguous()
    temps = torch.full((N,), float(temp))

    suffixes, rec = verify(
        logits_p=logits_p, logits_q=logits_q, speculations=speculations,
        temperatures_target=temps, temperatures_draft=temps,
        cache_hits=cache_hits, sampler_x=None, async_fan_out=None, jit_speculate=jit,
    )

    # Emitted token at the x1 slot: the draft token if accepted, else the recovery token.
    T = torch.tensor([(s[1] if len(s) == 2 else rec[b]) for b, s in enumerate(suffixes)])
    phat = torch.bincount(T, minlength=V).to(torch.float64) / N
    return phat, p0


def _noise_floor(p0, seed=1):
    torch.manual_seed(seed)
    ref = torch.multinomial(p0.expand(N, V), num_samples=1).squeeze(1)
    phat = torch.bincount(ref, minlength=V).to(torch.float64) / N
    return _tv(phat, p0)


@pytest.mark.parametrize("temp", [1.0, 0.7])
def test_sync_temp_gt0_is_distribution_preserving(temp):
    # Pre-fix: sync temp>0 skews toward argmax(p) (TV ~ 0.06, P(argmax): 0.40 -> 0.46).
    # Post-fix: the emitted law must equal p0 (TV down at the noise floor).
    phat, p0 = _run(temp, cache_hits=None, jit=False)
    floor = _noise_floor(p0)
    tv = _tv(phat, p0)
    assert tv < max(0.01, 5 * floor), (
        f"sync temp={temp}: TV={tv:.4f} floor={floor:.4f} phat={[round(x, 4) for x in phat.tolist()]}"
    )


def test_sync_temp0_greedy_unaffected():
    # Guard (temperature axis): temp==0 never reaches the ratio gate, so the fix must
    # leave greedy decoding exact.
    phat, p0 = _run(0.0, cache_hits=None, jit=False)
    floor = _noise_floor(p0)
    assert _tv(phat, p0) < max(0.01, 5 * floor)


def test_async_cache_hit_still_exact():
    # Guard (mode axis): async cache-hits already took the ratio path, so the fix must
    # leave them exact.
    phat, p0 = _run(1.0, cache_hits=torch.ones(N, dtype=torch.int64), jit=False)
    floor = _noise_floor(p0)
    assert _tv(phat, p0) < max(0.01, 5 * floor)


if __name__ == "__main__":
    print(f"{'cell':<28}{'TV':>10}{'floor':>10}   phat")
    for name, kw in [
        ("sync temp=1.0",       dict(temp=1.0, cache_hits=None, jit=False)),
        ("sync temp=0.7",       dict(temp=0.7, cache_hits=None, jit=False)),
        ("sync temp=0 (greedy)", dict(temp=0.0, cache_hits=None, jit=False)),
        ("async hit temp=1.0",  dict(temp=1.0, cache_hits=torch.ones(N, dtype=torch.int64), jit=False)),
    ]:
        phat, p0 = _run(**kw)
        tv, floor = _tv(phat, p0), _noise_floor(p0)
        print(f"{name:<28}{tv:>10.4f}{floor:>10.4f}   {[round(x, 4) for x in phat.tolist()]}")
