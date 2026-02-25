import torch
import torch.distributed as dist

from ssd.engine.sequence import Sequence
from ssd.utils.async_helpers.nccl_pack import send_int64, recv_int64


def send_speculation_request(
    seqs: list[Sequence],
    lookahead: int,
    async_fan_out: int,
    max_blocks: int,
    eagle: bool,
    draft_dtype: torch.dtype,
    device: torch.device,
    async_pg: dist.ProcessGroup,
    draft_runner_rank: int,
):
    """Send the complete request: cmd, metadata, and payload."""

    # Validate critical sequence state that's not obvious from type hints
    assert len(seqs) > 0, "seqs must be non-empty"
    for i, seq in enumerate(seqs):
        assert seq.recovery_token_id is not None, f"seq[{i}].recovery_token_id cannot be None - required for cache key generation"
        assert len(seq.draft_block_table) <= max_blocks, f"seq[{i}].draft_block_table length ({len(seq.draft_block_table)}) exceeds max_blocks ({max_blocks})"

    # Send spec request command
    cmd, meta, cache_keys, num_tokens, temperatures, draft_block_tables, recovery_activations = prepare_speculation_request_payload(
        seqs,
        len(seqs),
        lookahead,
        async_fan_out,
        device,
        max_blocks,
        eagle,
    )

    # Send spec request command (0 for spec)
    dist.send(cmd, dst=draft_runner_rank, group=async_pg)

    # Send metadata (B, K, F)
    dist.send(meta, dst=draft_runner_rank, group=async_pg)

    # Send payload data
    assert num_tokens.shape == (len(seqs),)
    send_int64(
        async_pg,
        draft_runner_rank,
        cache_keys,
        num_tokens,
        draft_block_tables,
    )
    # These are float tensors, so we need to send them separately
    dist.send(temperatures, dst=draft_runner_rank, group=async_pg)
    if recovery_activations is not None:
        # Cast to draft dtype to match draft receive buffer
        dist.send(recovery_activations.to(draft_dtype), dst=draft_runner_rank, group=async_pg)


def receive_speculation_response(
    batch_size: int,
    lookahead: int,
    vocab_size: int,
    draft_dtype: torch.dtype,
    device: torch.device,
    async_pg: dist.ProcessGroup,
    draft_runner_rank: int,
):
    """Receive the response: cache hits, speculations, and logits.

    Returns:
        speculations: [B, K] tensor of speculated tokens
        logits_q: [B, K, V] tensor of draft model logits
        cache_hits: [B] tensor of cache hit indicators
    """
    # Contract: recv_int64 expects exactly B + B*K elements for fused response
    B, K, V = batch_size, lookahead, vocab_size
    expected_fused_size = B + B * K
    fused_response = recv_int64(async_pg, draft_runner_rank, expected_fused_size, device)

    # Split fused response according to protocol contract
    cache_hits = fused_response[:B]  # [B]
    spec_tokens_flat = fused_response[B:]  # [B*K]

    # Reshape spec tokens according to protocol contract: flat [B*K] -> [B, K]
    speculations = spec_tokens_flat.view(B, K)

    # Draft now returns full target vocab size logits (after d2t expansion)
    logits_q = torch.empty(B, K, V, dtype=draft_dtype, device=device)
    dist.recv(logits_q, src=draft_runner_rank, group=async_pg)

    # Post-condition shape validation
    assert speculations.shape == (B, K), f"speculations shape mismatch: expected ({B}, {K}), got {speculations.shape}"
    assert logits_q.shape == (B, K, V), f"logits_q shape mismatch: expected ({B}, {K}, {V}), got {logits_q.shape}"

    return speculations, logits_q, cache_hits


def prepare_prefill_payload(
    input_id_list: list[list[int]],
    eagle_acts: torch.Tensor,
    device: torch.device,
    max_blocks: int,
    draft_block_tables: list[list[int]] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids_flat = []
    num_tokens = []
    for input_ids in input_id_list:
        input_ids_flat.extend(input_ids)
        num_tokens.append(len(input_ids))
    input_ids_flat = torch.tensor(input_ids_flat, dtype=torch.int64, device=device)
    num_tokens = torch.tensor(num_tokens, dtype=torch.int64, device=device)
    if isinstance(draft_block_tables, list):
        draft_block_table = torch.tensor(
            [dbt + [-1] * (max_blocks - len(dbt)) for dbt in draft_block_tables],
            dtype=torch.int32, device=device,
        )
    else:
        assert draft_block_tables.shape == (len(input_id_list), max_blocks), (
            f"draft_block_tables shape mismatch: expected ({len(input_id_list), max_blocks}), got {draft_block_tables.shape}"
        )
        draft_block_table = draft_block_tables

    # 3) send cmd=1
    cmd = torch.tensor([1], dtype=torch.int64, device=device)

    # 4) send metadata for tensor reconstruction
    metadata = torch.tensor([
        input_ids_flat.size(0),
        len(input_id_list),  # batch_size
        max_blocks,
        1 if eagle_acts is not None else 0,
        eagle_acts.shape[1] if eagle_acts is not None else 0,
    ], dtype=torch.int64, device=device)

    if eagle_acts is not None:
        assert eagle_acts.shape[0] == input_ids_flat.shape[0], (
            f"Eagle activations length {eagle_acts.shape[0]} != input_ids_flat length {input_ids_flat.shape[0]}"
        )

    return cmd, metadata, input_ids_flat, num_tokens, draft_block_table, eagle_acts


def prepare_speculation_request_payload(seqs, B, K, F, device, max_blocks, eagle):
    """Prepare handshake information for draft tree cache RPC."""
    # Build cache keys - shape contract: [B, 3] where columns are [seq_id, keep_idx, recovery_token]

    cmd = torch.tensor([0], dtype=torch.int64, device=device)
    meta = torch.tensor([B, K, F], dtype=torch.int64, device=device)

    # Build cache keys - shape contract: [B, 3] where columns are [seq_id, keep_idx, recovery_token]
    seq_ids = torch.tensor([s.seq_id for s in seqs], device=device)
    keep_idxs = torch.tensor([s.last_spec_step_accepted_len - 1 for s in seqs], device=device)
    recs = torch.tensor([s.recovery_token_id for s in seqs], device=device)
    cache_keys = torch.stack([seq_ids, keep_idxs, recs], dim=1)  # [B, 3]

    # Prepare num_tokens - shape contract: [B]
    num_tokens = torch.tensor(
        [seq.num_tokens for seq in seqs], dtype=torch.int64, device=device)  # [B]

    # Draft-side temperatures for tree decode: prefer per-seq override, else global config override, else seq.temperature
    temperatures = torch.tensor(
        [seq.draft_temperature if seq.draft_temperature is not None else seq.temperature for seq in seqs],
        dtype=torch.float32,
        device=device,
    )  # [B]

    # Prepare draft block tables - shape contract: [B, max_blocks] with -1 padding
    draft_block_tables = torch.tensor(
        [seq.draft_block_table + [-1] * (max_blocks - len(seq.draft_block_table)) for seq in seqs],
        dtype=torch.int64,
        device=device,
    )  # [B, max_blocks]

    # Prepare recovery activations for EAGLE
    if eagle:
        for i, seq in enumerate(seqs):
            assert seq.last_target_hidden_state is not None, \
                f"seq[{i}].last_target_hidden_state is None - must be set after prefill/verify"
        recovery_activations = torch.stack(
            [seq.last_target_hidden_state for seq in seqs],
            dim=0,
        ).to(device)
    else:
        recovery_activations = None

    # Post-condition shape validation
    assert cache_keys.shape == (B, 3), f"cache_keys shape mismatch: expected ({B}, 3), got {cache_keys.shape}"
    assert num_tokens.shape == (B,), f"num_tokens shape mismatch: expected ({B},), got {num_tokens.shape}"
    assert temperatures.shape == (B,), f"temperatures shape mismatch: expected ({B},), got {temperatures.shape}"
    assert draft_block_tables.shape == (B, max_blocks), f"draft_block_tables shape mismatch: expected ({B}, {max_blocks}), got {draft_block_tables.shape}"

    return cmd, meta, cache_keys, num_tokens, temperatures, draft_block_tables, recovery_activations

def prepare_decode_tensors_from_seqs(
    seqs: list[Sequence],
    block_size: int,
    is_draft: bool,
    verify: bool = False,
    k: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []

    if not verify:  # normal decoding or draft fwd in speculation
        assert k == -1, "k should be -1 for normal decoding or draft fwd in speculation"
        for seq in seqs:
            block_table = seq.draft_block_table if is_draft else seq.block_table
            assert len(seq) // block_size <= len(block_table), "in sync spec draft decode, not enough blocks allocated"
            num_cached_tokens = seq.num_cached_tokens if not is_draft else seq.num_draft_cached_tokens
            assert num_cached_tokens == len(seq) - 1, "num_cached_tokens should be equal to len(seq) - 1 in pure sq decode path"
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))

            pos = seq.num_tokens - 1
            block_idx = pos // block_size
            pos_in_block = pos % block_size
            slot_mapping.append(block_table[block_idx] * block_size + pos_in_block)
    else:  # verify and glue decode prep both go here
        assert not is_draft, "verify path only supported for target model" # we prep tensors to send to draft for glue on the target 
        assert k > 0, "k should be > 0 for target fwd in verify"

        for seq_idx, seq in enumerate(seqs):
            # can hardcode block_table here for target since this is only target codepath 
            assert (seq.num_tokens - 1) // block_size <= len(seq.block_table), "in sync spec target verify, not enough blocks allocated"
            
            pos0 = seq.num_tokens - (k+1)
            input_ids.extend(seq[pos0:])
            positions.extend(list(range(pos0, pos0 + k + 1)))
            assert seq.num_cached_tokens == pos0, f"num_cached_tokens={seq.num_cached_tokens} != pos0={pos0} (num_tokens={seq.num_tokens}, k={k})"
            context_lens.append(len(seq))  

            for j in range(k + 1):
                pos = pos0 + j
                block_idx = pos // block_size
                block_id = seq.block_table[block_idx]
                pos_in_block = pos % block_size
                slot_mapping.append(
                    block_id * block_size + pos_in_block)


    input_ids = torch.tensor(
        input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(
        positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(
        slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens = torch.tensor(
        context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    return input_ids, positions, slot_mapping, context_lens

def prepare_block_tables_from_seqs(
    seqs: list[Sequence],
    is_draft: bool = False
) -> torch.Tensor:
        if is_draft:
            max_len = max(len(seq.draft_block_table) for seq in seqs)
            block_tables = [seq.draft_block_table + [-1] * (max_len - len(seq.draft_block_table)) for seq in seqs]
        else:
            max_len = max(len(seq.block_table) for seq in seqs)
            block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

def prepare_prefill_tensors_from_seqs(
    seqs: list[Sequence],
    block_size: int,
    is_draft: bool = False,
    skip_first_token: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert skip_first_token in (0, 1)
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    
    for seq in seqs:
        seqlen = len(seq)
        if is_draft:
            num_cached_tokens = seq.num_draft_cached_tokens
            block_table = seq.draft_block_table
        else:
            num_cached_tokens = seq.num_cached_tokens
            block_table = seq.block_table

        start = num_cached_tokens + (skip_first_token if is_draft else 0)
        input_ids.extend(seq[start:])
        pos_offset = -skip_first_token if is_draft else 0
        positions.extend(list(range(start + pos_offset, seqlen + pos_offset)))
        seqlen_q = seqlen - start
        seqlen_k = seqlen + pos_offset
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)

        if not block_table:  # first prefill
            continue

        # new: emit exactly one slot for each *new* token
        #    map each token index -> (block_id * block_size + offset)
        for pos in range(start + pos_offset, seq.num_tokens + pos_offset):
            block_i = pos // block_size
            offset = pos % block_size
            slot = block_table[block_i] * block_size + offset
            slot_mapping.append(slot)

    input_ids = torch.tensor(
        input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(
        positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_q = torch.tensor(
        cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_k = torch.tensor(
        cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(
        slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    
    return input_ids, positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping


