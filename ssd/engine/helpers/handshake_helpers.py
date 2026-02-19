import torch
import torch.distributed as dist

from ssd.engine.sequence import Sequence
from ssd.utils.async_helpers.nccl_pack import send_int64, recv_int64


class TargetDraftHandshake:
    """Pre-allocated handshake handler. Created once at init, reused every step."""

    def __init__(
        self,
        lookahead: int,
        async_fan_out: int,
        max_blocks: int,
        eagle: bool,
        vocab_size: int,
        draft_dtype: torch.dtype,
        device: torch.device,
        async_pg: dist.ProcessGroup,
        draft_runner_rank: int,
    ):
        self.K = lookahead
        self.F = async_fan_out
        self.max_blocks = max_blocks
        self.eagle = eagle
        self.vocab_size = vocab_size
        self.draft_dtype = draft_dtype
        self.device = device
        self.async_pg = async_pg
        self.draft_runner_rank = draft_runner_rank

        # Pre-allocate send buffers (B=1, reallocated if B changes)
        self._alloc_buffers(1)

    def _alloc_buffers(self, B):
        self.B = B
        d = self.device
        # Send: constant per-step
        self.cmd = torch.zeros(1, dtype=torch.int64, device=d)  # 0 = spec
        self.meta = torch.tensor([B, self.K, self.F], dtype=torch.int64, device=d)
        # Send: variable per-step (filled in-place)
        self.cache_keys = torch.empty(B, 3, dtype=torch.int64, device=d)
        self.num_tokens_buf = torch.empty(B, dtype=torch.int64, device=d)
        self.temps_buf = torch.empty(B, dtype=torch.float32, device=d)
        self.block_tables_buf = torch.full((B, self.max_blocks), -1, dtype=torch.int32, device=d)
        # Pre-allocate int64 conversion buffers (avoid per-step allocation)
        self._block_tables_int64 = torch.empty(B, self.max_blocks, dtype=torch.int64, device=d)
        self._temps_int64 = torch.empty(B, dtype=torch.int64, device=d)
        # Recv: pre-allocated
        self.fused_response = torch.empty(B + B * self.K, dtype=torch.int64, device=d)
        self.logits_q = torch.empty(B, self.K, self.vocab_size, dtype=self.draft_dtype, device=d)

    def execute(self, seqs: list[Sequence]):
        """Fill buffers, send request, receive response."""
        B = len(seqs)
        if B != self.B:
            self._alloc_buffers(B)

        # Fill send buffers in-place (avoids torch.tensor from Python lists)
        for i, seq in enumerate(seqs):
            self.cache_keys[i, 0] = seq.seq_id
            self.cache_keys[i, 1] = seq.last_spec_step_accepted_len - 1
            self.cache_keys[i, 2] = seq.recovery_token_id
            self.num_tokens_buf[i] = seq.num_tokens
            self.temps_buf[i] = seq.draft_temperature or seq.temperature
            bt = seq.draft_block_table
            bt_len = len(bt)
            if bt_len > 0:
                self.block_tables_buf[i, :bt_len] = torch.tensor(bt, dtype=torch.int32, device=self.device)
            self.block_tables_buf[i, bt_len:] = -1

        # Send cmd + meta + fused payload
        dist.send(self.cmd, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self.meta, dst=self.draft_runner_rank, group=self.async_pg)
        self._block_tables_int64.copy_(self.block_tables_buf)
        self._temps_int64.copy_(self.temps_buf.view(torch.int32).to(torch.int64))
        send_int64(
            self.async_pg, self.draft_runner_rank,
            self.cache_keys, self.num_tokens_buf,
            self._block_tables_int64, self._temps_int64,
        )

        if self.eagle:
            recovery_activations = torch.stack([
                seq.last_target_hidden_state for seq in seqs
            ], dim=0).to(self.device)
            dist.send(recovery_activations.to(self.draft_dtype), dst=self.draft_runner_rank, group=self.async_pg)

        # Recv into pre-allocated buffers
        self._skip_logits = all(seq.temperature == 0 and (seq.draft_temperature is None or seq.draft_temperature == 0) for seq in seqs)
        dist.recv(self.fused_response, src=self.draft_runner_rank, group=self.async_pg)
        cache_hits = self.fused_response[:B]
        speculations = self.fused_response[B:].view(B, self.K)
        if not self._skip_logits:
            dist.recv(self.logits_q, src=self.draft_runner_rank, group=self.async_pg)

        return speculations, self.logits_q, cache_hits
