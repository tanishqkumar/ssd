import torch 
import torch.distributed as dist

from ssd.engine.sequence import Sequence
from ssd.utils.async_helpers.nccl_pack import send_int64, recv_int64

class TargetDraftHandshake:
    """Handles the complete handshake protocol between target and draft runners."""
    
    def __init__(self, seqs: list[Sequence], config, async_pg, draft_runner_rank):
        # Input validation for non-obvious requirements
        assert len(seqs) > 0, "seqs must be non-empty"
        assert async_pg is not None, "async_pg (process group) cannot be None"
        
        self.seqs = seqs
        self.config = config
        self.async_pg = async_pg
        self.draft_runner_rank = draft_runner_rank
        self.device = config.device
        self.B = len(seqs)
        self.K = config.speculate_k
        self.F = config.async_fan_out
        
        # Validate critical sequence state that's not obvious from type hints
        for i, seq in enumerate(seqs):
            assert seq.recovery_token_id is not None, f"seq[{i}].recovery_token_id cannot be None - required for cache key generation"
            assert len(seq.draft_block_table) <= config.max_blocks, f"seq[{i}].draft_block_table length ({len(seq.draft_block_table)}) exceeds max_blocks ({config.max_blocks})"
        
        # Prepare payload data
        self._prepare_payload()
    
    def _prepare_payload(self):
        """Prepare handshake information for draft tree cache RPC."""
        # Build cache keys - shape contract: [B, 3] where columns are [seq_id, keep_idx, recovery_token]
        seq_ids = torch.tensor([s.seq_id for s in self.seqs], device=self.device)
        keep_idxs = torch.tensor([s.last_spec_step_accepted_len - 1 for s in self.seqs], device=self.device) 
        recs = torch.tensor([s.recovery_token_id for s in self.seqs], device=self.device)
        self.cache_keys = torch.stack([seq_ids, keep_idxs, recs], dim=1)  # [B, 3]
        
        # Prepare num_tokens - shape contract: [B]
        self.num_tokens = torch.tensor(
            [seq.num_tokens for seq in self.seqs], dtype=torch.int64, device=self.device)  # [B]
        
        # Draft-side temperatures for tree decode: prefer per-seq override, else global config override, else seq.temperature
        temps_draft = []
        for seq in self.seqs:
            if getattr(seq, 'draft_async_temperature', None) is not None:
                temps_draft.append(seq.draft_async_temperature)
            elif self.config.draft_async_temp is not None:
                temps_draft.append(self.config.draft_async_temp)
            else:
                temps_draft.append(seq.temperature)

        self.temperatures = torch.tensor(temps_draft, dtype=torch.float32, device=self.device)  # [B]

        # Prepare draft block tables - shape contract: [B, max_blocks] with -1 padding
        self.draft_block_tables = torch.tensor([seq.draft_block_table + [-1] * (  
            self.config.max_blocks - len(seq.draft_block_table)) for seq in self.seqs], dtype=torch.int32, device=self.device)  # [B, max_blocks]
        
        # Post-condition shape validation
        assert self.cache_keys.shape == (self.B, 3), f"cache_keys shape mismatch: expected ({self.B}, 3), got {self.cache_keys.shape}"
        assert self.num_tokens.shape == (self.B,), f"num_tokens shape mismatch: expected ({self.B},), got {self.num_tokens.shape}"
        assert self.temperatures.shape == (self.B,), f"temperatures shape mismatch: expected ({self.B},), got {self.temperatures.shape}"
        assert self.draft_block_tables.shape == (self.B, self.config.max_blocks), f"draft_block_tables shape mismatch: expected ({self.B}, {self.config.max_blocks}), got {self.draft_block_tables.shape}"
    
    def send_request(self):
        """Send the complete request: cmd, metadata, and payload."""
        # Send spec request command
        cmd = torch.tensor([0], dtype=torch.int64, device=self.device)  # 0 for spec
        dist.send(cmd, dst=self.draft_runner_rank, group=self.async_pg)
        
        # Send metadata (B, K, F)
        meta = torch.tensor([self.B, self.K, self.F], dtype=torch.int64, device=self.device)
        dist.send(meta, dst=self.draft_runner_rank, group=self.async_pg)

        # Send payload data
        assert self.num_tokens.shape == (self.B,)
        send_int64(
            self.async_pg,
            self.draft_runner_rank,
            self.cache_keys,
            self.num_tokens,
            self.draft_block_tables.to(torch.int64),
        )
        dist.send(self.temperatures, dst=self.draft_runner_rank, group=self.async_pg)
    
    def receive_response(self, draft_cfg):
        """Receive the response: cache hits, speculations, and logits.
        
        Returns:
            cache_hits: [B] tensor of cache hit indicators
            speculations: [B, K] tensor of speculated tokens  
            logits_q: [B, K, V] tensor of draft model logits
        """
        # Contract: recv_int64 expects exactly B + B*K elements for fused response
        expected_fused_size = self.B + self.B * self.K
        fused_response = recv_int64(self.async_pg, self.draft_runner_rank, expected_fused_size, self.device)
        
        # Split fused response according to protocol contract
        cache_hits = fused_response[:self.B]  # [B]
        spec_tokens_flat = fused_response[self.B:]  # [B*K]

        # Reshape spec tokens according to protocol contract: flat [B*K] -> [B, K]
        speculations = spec_tokens_flat.view(self.B, self.K)
        
        # Receive logits - shape contract: [B, K, V] where V is vocab_size
        assert hasattr(draft_cfg, 'hf_config'), "draft_cfg must have hf_config attribute"
        assert hasattr(draft_cfg.hf_config, 'vocab_size'), "draft_cfg.hf_config must have vocab_size attribute"
        assert hasattr(draft_cfg.hf_config, 'torch_dtype'), "draft_cfg.hf_config must have torch_dtype attribute"
        
        V = draft_cfg.hf_config.vocab_size
        logits_q = torch.empty(self.B, self.K, V, dtype=draft_cfg.hf_config.torch_dtype, device=self.device)
        dist.recv(logits_q, src=self.draft_runner_rank, group=self.async_pg)
        
        # Post-condition shape validation
        assert cache_hits.shape == (self.B,), f"cache_hits shape mismatch: expected ({self.B},), got {cache_hits.shape}"
        assert speculations.shape == (self.B, self.K), f"speculations shape mismatch: expected ({self.B}, {self.K}), got {speculations.shape}"
        assert logits_q.shape == (self.B, self.K, V), f"logits_q shape mismatch: expected ({self.B}, {self.K}, {V}), got {logits_q.shape}"
        
        return cache_hits, speculations, logits_q
    
    def execute_full_handshake(self, draft_cfg):
        """Execute the complete handshake: send request and receive response."""
        self.send_request()
        return self.receive_response(draft_cfg)