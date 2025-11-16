import torch

def verify(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    speculations: torch.Tensor,
    temperatures_target: torch.Tensor,
    temperatures_draft: torch.Tensor | None = None,
) -> tuple[list[list[int]], list[int]]:
    """
    Speculative‐decoding verification:
     - For temp==0: pure argmax‐compare (greedy).
     - For temp>0: softmax + p/q‐ratio acceptance + (re)sampling.
    
    IMPORTANT: logits_p should be the target model's logits for positions
    where the draft tokens were generated, NOT the positions after them.
    For example, if draft generated tokens at positions L, L+1, ..., L+K-1,
    then logits_p should be the target's logits at those same positions,
    i.e., indices L-1, L, ..., L+K-2 in the full sequence.
    """
    device = logits_p.device
    B, K, V = logits_p.shape

    # Default draft temperatures to zero if not provided
    if temperatures_draft is None:
        temperatures_draft = torch.zeros_like(temperatures_target)

    # 1) Greedy argmax paths (for all, we precompute preds_p)
    # ------------------------------------------------------
    # draft_tokens[b,j] = speculations[b, j] = x_{j}
    draft_tokens = speculations                              # [B, K]
    # preds_p[b,i] = argmax on logits_p[b,i] => p_{i} argmax
    preds_p = logits_p.argmax(dim=-1)                        # [B, K]

    # Compare x_j against preds_p[:, j] for j=0..K-1
    matches = draft_tokens == preds_p                        # [B, K]
    any_mismatch = (~matches).any(dim=1)                     # [B]
    first_mismatch = (~matches).int().argmax(dim=1)          # [B]
    # accept up to K if no mismatch, else up to first_mismatch
    accept_greedy = torch.where(
        any_mismatch,
        first_mismatch,
        torch.full_like(first_mismatch, K)
    )                                                        # [B]
    batch_idx = torch.arange(B, device=device)
    # greedy recovery = preds_p[b, accept_greedy[b]] if accept_greedy[b] < K, else None
    # IMPORTANT: Avoid out-of-bounds by clamping indices before advanced indexing
    valid_greedy = (accept_greedy < K)
    safe_greedy_idx = accept_greedy.clamp(max=K-1)
    preds_at_safe = preds_p[batch_idx, safe_greedy_idx]
    rec_greedy = torch.where(
        valid_greedy,
        preds_at_safe,
        torch.full_like(accept_greedy, -1)  # placeholder for no recovery needed
    )                                                        # [B]

    # 2) Ratio‐based acceptance (only needed if any temp>0)
    # ------------------------------------------------------
    temps_t = temperatures_target
    temps_q = temperatures_draft

    # Rows eligible for ratio-acceptance must have any temp>0
    ratio_rows = ((temps_t > 0) | (temps_q > 0))
    do_any_ratio = ratio_rows.any().item()

    # We need probs_p for recovery sampling whenever any temps_t>0 exists
    need_p_probs = (temps_t > 0).any().item() or do_any_ratio

    # Prepare probability tensors as needed
    probs_p = None
    if need_p_probs:
        probs_p = torch.zeros(B, K, V, device=device, dtype=torch.float32)
        nz_p = (temps_t > 0)
        if nz_p.any():
            t = temps_t[nz_p].unsqueeze(1).unsqueeze(2).clamp(min=1e-8)
            probs_p[nz_p] = torch.softmax(
                (logits_p[nz_p] / t).to(torch.float32), dim=-1)
        z_p = (~nz_p)
        if z_p.any():
            argmax_p = logits_p[z_p].argmax(dim=-1)  # [Bz, K]
            one_hot_p = torch.zeros_like(logits_p[z_p], dtype=torch.float32)
            one_hot_p.scatter_(2, argmax_p.unsqueeze(-1), 1.0)
            probs_p[z_p] = one_hot_p

    # Ratio acceptance path (only for ratio_rows)
    if do_any_ratio:
        probs_q = torch.zeros(B, K, V, device=device, dtype=torch.float32)
        nz_q = (temps_q > 0)
        if nz_q.any():
            tq = temps_q[nz_q].unsqueeze(1).unsqueeze(2).clamp(min=1e-8)
            probs_q[nz_q] = torch.softmax(
                (logits_q[nz_q] / tq).to(torch.float32), dim=-1)
        z_q = (~nz_q)
        if z_q.any():
            argmax_q = logits_q[z_q].argmax(dim=-1)  # [Bz, K]
            one_hot_q = torch.zeros_like(logits_q[z_q], dtype=torch.float32)
            one_hot_q.scatter_(2, argmax_q.unsqueeze(-1), 1.0)
            probs_q[z_q] = one_hot_q

        # gather p_i(x_i) and q_i(x_i) for i=0..K-1
        p_all = probs_p
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
    #  - Else: sample directly from p
    # For rows with temps_t==0: use greedy
    batch_idx = torch.arange(B, device=device)
    if probs_p is None:
        # No temperatures require sampling; keep greedy
        rec_ratio = rec_greedy
    else:
        # For recovery, we need to sample from position accept_until when accept_until < K
        need_recovery = accept_until < K

        # Clamp indices before any advanced indexing to avoid device-side assert
        recovery_positions = accept_until.clamp(max=K-1)
        p_fallback = probs_p[batch_idx, recovery_positions]  # [B, V]
        p_sum = p_fallback.sum(dim=1, keepdim=True).clamp(min=1e-12)
        fallbackDist = p_fallback / p_sum

        if do_any_ratio:
            q_slice = probs_q[batch_idx, recovery_positions]  # [B, V]
            mask_adjust = (temps_t > 0) & need_recovery & ratio_rows

            # Adjusted distribution: max(0, p - q). If it has no mass, fall back to p
            adj = (p_fallback - q_slice).clamp(min=0.0)       # [B, V]
            adj_sum = adj.sum(dim=1, keepdim=True)            # [B, 1]
            has_mass = adj_sum > 0
            adjusted_dist = torch.where(
                has_mass, adj / adj_sum, fallbackDist
            )                                                 # [B, V]

            rec_ratio_adjusted = torch.multinomial(adjusted_dist, 1).squeeze(1)
            rec_from_p = torch.multinomial(fallbackDist, 1).squeeze(1)
            rec_ratio = torch.where(mask_adjust, rec_ratio_adjusted, rec_from_p)
        else:
            # No ratio rows; sample from p for temps_t>0
            rec_ratio = torch.multinomial(fallbackDist, 1).squeeze(1)

    # final recovery tokens (only valid when accept_until < K)
    rec_final = torch.where(
        (temps_t > 0) & (accept_until < K),
        rec_ratio,
        torch.where(accept_until < K, rec_greedy, torch.full_like(rec_greedy, -1))
    )  # [B]

    # 4) Materialize ragged accepted_suffixes
    # ---------------------------------------
    # For K>1, accept_until can be 0..K inclusive semantics; we clamp to [0..K]
    accept_until_clamped = accept_until.clamp(min=0, max=K)
    counts = accept_until_clamped.tolist()
    accepted_suffixes: list[list[int]] = []
    for b in range(B):
        n = counts[b]
        n = int(max(0, min(K, n)))
        suffix = draft_tokens[b, :n].tolist()
        accepted_suffixes.append(suffix)

    return accepted_suffixes, rec_final.tolist()
