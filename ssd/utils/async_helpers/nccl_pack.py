import torch
import torch.distributed as dist


def concat_int64(*tensors: torch.Tensor) -> torch.Tensor:
    """Concatenate tensors into a single flat int64 payload."""
    parts = []
    for t in tensors:
        if t is None:
            continue
        if t.dtype != torch.int64:
            t = t.to(torch.int64)
        parts.append(t.reshape(-1))
    if not parts:
        return torch.empty(0, dtype=torch.int64)
    return torch.cat(parts, dim=0)


def send_int64(pg, dst: int, *tensors: torch.Tensor):
    """Send many int64-compatible tensors as one fused payload in a fixed order."""
    payload = concat_int64(*tensors)
    if payload.numel() == 0:
        return
    dist.send(payload, dst=dst, group=pg)


def recv_int64(pg, src: int, total_length: int, device: torch.device) -> torch.Tensor:
    """Receive a fused int64 payload of known total length."""
    t = torch.empty((total_length,), dtype=torch.int64, device=device)
    if total_length > 0:
        dist.recv(t, src=src, group=pg)
    return t


