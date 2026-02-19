import torch
from ssd.utils.async_helpers.async_spec_helpers import apply_sampler_x_rescaling
from ssd.config import Config

def verify(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    speculations: torch.Tensor,
    temperatures_target,
    temperatures_draft,
    cache_hits: torch.Tensor | None = None,
    sampler_x: float | None = None,
    async_fan_out: int | None = None,
    jit_speculate: bool = False,
    all_greedy: bool = False,
) -> tuple[list[list[int]], list[int]]:
    """
    Speculative‐decoding verification:
     - For temp==0: pure argmax‐compare (greedy).
     - For temp>0: softmax + p/q‐ratio acceptance + (re)sampling.
     - IMPORTANT: Only apply ratio acceptance on rows where the draft proposal truly
       came from q (cache hit in async mode). On cache misses, fall back to greedy
       acceptance and sample recovery directly from p.
    """

    device = logits_p.device
    B, Kp1, V = logits_p.shape
    K = Kp1 - 1

    # FAST PATH: all temps == 0, single GPU->CPU sync
    if all_greedy:
        _draft = speculations[:, 1:]
        _preds = logits_p.argmax(dim=-1)
        _match = (_draft == _preds[:, :-1]).int()
        _packed = torch.cat([speculations[:, 0], _preds.reshape(-1),
                             _draft.reshape(-1), _match.reshape(-1)])
        _cpu = _packed.cpu().tolist()
        _suffixes, _recs = [], []
        for _b in range(B):
            _s = _cpu[_b]
            _p = _cpu[B + _b * Kp1: B + (_b + 1) * Kp1]
            _d = _cpu[B + B * Kp1 + _b * K: B + B * Kp1 + (_b + 1) * K]
            _m = _cpu[B + B * Kp1 + B * K + _b * K: B + B * Kp1 + B * K + (_b + 1) * K]
            _n = 0
            while _n < K and _m[_n]:
                _n += 1
            _suffixes.append([_s] + _d[:_n])
            _recs.append(int(_p[_n]))
        return _suffixes, _recs

    # 1) Greedy argmax paths (for all, we precompute preds_p)
    # ------------------------------------------------------
    # draft_tokens[b,j] = speculations[b, j+1] = x_{j+1}
    draft_tokens = speculations[:, 1:]                   # [B, K]
    # preds_p[b,i] = argmax on logits_p[b,i] => p_{i+1} argmax
    preds_p = logits_p.argmax(dim=-1)                    # [B, K+1]

    # Compare x_j against preds_p[:, j] for j=0..K-1
    matches = draft_tokens == preds_p[:, :-1]            # [B, K]
    any_mismatch = (~matches).any(dim=1)                 # [B]
    first_mismatch = (~matches).int().argmax(dim=1)      # [B]
    # accept up to K if no mismatch, else up to first_mismatch
    accept_greedy = torch.where(
        any_mismatch,
        first_mismatch,
        torch.full_like(first_mismatch, K)
    )                                                    # [B]
    batch_idx = torch.arange(B, device=device)
    # greedy recovery = preds_p[b, accept_greedy[b]]
    rec_greedy = preds_p[batch_idx, accept_greedy]       # [B]

    # 2) Ratio‐based acceptance (only needed if any temp>0)
    # ------------------------------------------------------
    temps_t = temperatures_target
    temps_q = temperatures_draft

    # Rows eligible for ratio-acceptance must both need ratio (any temp>0)
    # AND be cache hits (i.e., tokens were actually sampled from q).
    base_ratio_rows = ((temps_t > 0) | (temps_q > 0))
    
    # TODO: understanding why we gate on cache_hits is important, maybe cause of our --x not working
    if jit_speculate:
        ratio_rows = base_ratio_rows
    else:
        ratio_rows = base_ratio_rows & (cache_hits.to(torch.bool) if cache_hits is not None else torch.zeros_like(base_ratio_rows, dtype=torch.bool))

    do_any_ratio = ratio_rows.any().item()

    # We need probs_p for recovery sampling whenever any temps_t>0 exists,
    # regardless of whether we end up doing ratio on any row.
    need_p_probs = (temps_t > 0).any().item() or do_any_ratio

    # Prepare probability tensors as needed
    B, Kp1, V = logits_p.shape
    K = Kp1 - 1

    probs_p = None
    if need_p_probs:
        probs_p = torch.zeros(B, Kp1, V, device=device, dtype=torch.float32)
        nz_p = (temps_t > 0)
        if nz_p.any():
            t = temps_t[nz_p].unsqueeze(1).unsqueeze(2).clamp(min=1e-8)
            probs_p[nz_p] = torch.softmax((logits_p[nz_p] / t).to(torch.float32), dim=-1)
        z_p = (~nz_p)
        if z_p.any():
            argmax_p = logits_p[z_p].argmax(dim=-1)  # [Bz, K+1]
            one_hot_p = torch.zeros_like(logits_p[z_p], dtype=torch.float32)
            one_hot_p.scatter_(2, argmax_p.unsqueeze(-1), 1.0)
            probs_p[z_p] = one_hot_p

    # Ratio acceptance path (only for ratio_rows)
    if do_any_ratio:
        probs_q = torch.zeros(B, K, V, device=device, dtype=torch.float32)
        nz_q = (temps_q > 0)
        if nz_q.any():
            tq = temps_q[nz_q].unsqueeze(1).unsqueeze(2).clamp(min=1e-8)
            probs_q[nz_q] = torch.softmax((logits_q[nz_q] / tq).to(torch.float32), dim=-1)
        z_q = (~nz_q)
        if z_q.any():
            argmax_q = logits_q[z_q].argmax(dim=-1)  # [Bz, K]
            one_hot_q = torch.zeros_like(logits_q[z_q], dtype=torch.float32)
            one_hot_q.scatter_(2, argmax_q.unsqueeze(-1), 1.0)
            probs_q[z_q] = one_hot_q

        # Apply sampler_x rescaling to draft distribution if provided
        if sampler_x is not None:
            assert async_fan_out is not None, "async_fan_out must be provided if sampler_x is provided"
            probs_q = apply_sampler_x_rescaling(probs_q, sampler_x, async_fan_out)

        # gather p_i(x_i) and q_i(x_i) for i=1..K on rows doing ratio
        p_all = probs_p[:, :K, :] if probs_p is not None else torch.zeros(B, K, V, device=device, dtype=torch.float32)
        q_all = probs_q
        gather_idx = draft_tokens.unsqueeze(2)  # [B, K, 1]
        p_vals = p_all.gather(2, gather_idx).squeeze(2)  # [B, K]
        q_vals = q_all.gather(2, gather_idx).squeeze(2)  # [B, K]

        accept_probs = (p_vals / (q_vals + 1e-10)).clamp(max=1.0)  # [B, K]
        rand = torch.rand_like(accept_probs)
        accepts = rand <= accept_probs  # [B, K]

        rej_any = (~accepts).any(dim=1)  # [B]
        first_rej = (~accepts).int().argmax(dim=1)  # [B]
        accept_ratio = torch.where(
            rej_any,
            first_rej,
            torch.full_like(first_rej, K)
        )  # [B]

        # Only use ratio accept on ratio_rows; others fall back to greedy
        accept_until = torch.where(ratio_rows, accept_ratio, accept_greedy)
    else:
        # No rows use ratio; all fall back to greedy accept counts
        accept_until = accept_greedy

    # 3) Construct the recovery distribution and sample
    # For rows with temps_t>0:
    #  - If ratio_rows: use adjusted max(0, p - q) when accept<K, else p
    #  - Else (misses): sample directly from p
    # For rows with temps_t==0: use greedy
    batch_idx = torch.arange(B, device=device)
    if probs_p is None:
        # No temperatures require sampling; keep greedy
        rec_ratio = rec_greedy
    else:
        p_fallback = probs_p[batch_idx, accept_until]  # [B, V]
        p_sum = p_fallback.sum(dim=1, keepdim=True)
        fallbackDist = p_fallback / p_sum

        if do_any_ratio:
            p_all = probs_p[:, :K, :]
            q_idx_safe = accept_until.clamp(max=K-1)
            # Build q_all for adjusted rows only; reuse previous q_all if available
            # We already have q_all if do_any_ratio=True (and it's already rescaled if sampler_x was provided)
            q_slice = q_all[batch_idx, q_idx_safe]  # [B, V]
            mask_adjust = (temps_t > 0) & (accept_until < K) & ratio_rows

            adj = (p_fallback - q_slice).clamp(min=0.0)
            sums = adj.sum(dim=1, keepdim=True)
            adj_norm = torch.where(sums > 0, adj / sums, fallbackDist)
            # For ratio_rows use adj_norm when adjust, else fallback; for non-ratio rows with temp>0 use fallback; temp==0 use greedy
            rec_ratio_adjusted = torch.multinomial(adj_norm, 1).squeeze(1)
            rec_from_p = torch.multinomial(fallbackDist, 1).squeeze(1)
            rec_ratio = torch.where(mask_adjust, rec_ratio_adjusted, rec_from_p)
        else:
            # No ratio rows; sample from p for temps_t>0, else greedy
            rec_from_p = torch.multinomial(fallbackDist, 1).squeeze(1)
            rec_ratio = rec_from_p

    # final recovery tokens
    rec_final = torch.where(temps_t > 0, rec_ratio, rec_greedy)      # [B]

    # 4) Materialize ragged accepted_suffixes
    # ---------------------------------------
    accepted_suffixes: list[list[int]] = []
    # previous recovery
    starts = speculations[:, 0].tolist()
    counts = accept_until.tolist()

    for b in range(B):
        n = counts[b]
        suffix = [starts[b]] + draft_tokens[b, :n].tolist()
        accepted_suffixes.append(suffix)

    return accepted_suffixes, rec_final.tolist()
