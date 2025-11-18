# EAGLE-3 (Paper + vLLM) — Minimal, Inference-Focused, Linear Drafting Only

This summarizes exactly what happens at **inference**, assuming:
- **Target LLM on GPU₀**  
- **Draft model on GPU₁**  
- **Linear multi-token drafting** (no tree)

---

# 1. Data Flow per Round

Let current accepted prefix = tokens **x₁…xₜ**.

## (A) Target forward on GPU₀
1. Run target LLM on prefix (normal vLLM step).  
2. Collect:
   - next-token logits  
   - **multi-layer hidden states** for every prefix token  
   - KV cache (stays on GPU₀)

3. Extract **low/mid/high** hidden states per token → shape `[num_tokens, 3*h_target]`.  
4. These are sent to the draft side as `target_hidden_states`.  
5. The next real target token is sampled: `next_token_ids`.

---

# 2. Draft Model Inputs on GPU₁

The draft model receives:

- `input_ids`:  
  prefix tokens with the **last token replaced** by `next_token_ids`.  
- `positions`: positional indices of these tokens.  
- `target_hidden_states`: multi-layer features for the **entire prefix**.

### 2.1 Feature fusion
On GPU₁:
fused_g = fc(concat(low, mid, high)) # shape: [num_tokens, h_draft]

`fused_g[i]` is the feature **gᵢ** for prefix token i.

These fill the draft model’s internal `hidden_states` buffer for prefix positions.

### 2.2 Draft input representation
For each sequence position i:
embed = token_embedding(input_ids[i]) # ℝ^{h_draft}
feat = (gᵢ for prefix) or (aᵢ for drafted) # ℝ^{h_draft}
concat = [embed ; feat] # ℝ^{2*h_draft}

Only the **first draft layer** consumes this concatenation  
(via an enlarged QKV projection).

---

# 3. Drafting on GPU₁ (Linear K-step)

Repeat for step k = 1…K:

1. Run the draft transformer with:
   - current `input_ids`
   - current `positions`
   - current `hidden_states` (mix of gᵢ and previously produced aᵢ)

2. From the last position of each sequence, take:
h_last = last_hidden_state # ℝ^{h_draft}
logits = lm_head(h_last)
draft_token = sample(logits)

3. Append `draft_token` to the output sequence.

4. Update draft model state:
a_last = h_last # self-feature
hidden_states.append(a_last)
input_ids = draft_token
positions += 1

Prefix positions keep using fused gᵢ; only new positions use self-features aᵢ.

After K iterations, GPU₁ returns a `[batch_size, K]` tensor of draft tokens.

---

# 4. Verification on GPU₀ (Linear Speculative Check)

For each draft token in order:

1. Run the target model one step using GPU₀ KV cache.  
2. Compare draft token with target’s predicted distribution.  
3. Accept until the first disagreement (lossless speculative rule).  
4. Commit the accepted tokens to the prefix.

---

# 5. Next Round

1. New prefix = old prefix + committed tokens.  
2. Target forward again on GPU₀ → extract new low/mid/high states → fuse on GPU₁.  
3. Run draft K-step rollout again.  
4. Repeat.

---

# 6. Essential Points

- **All prefix tokens** supply fused features gᵢ.  
- **All drafted tokens** use self-generated features aᵢ.  
- Fused features come from the multi-layer target hidden states via one FC.  
- Draft model runs **entirely** on GPU₁; verification runs **entirely** on GPU₀.  
- Cross-GPU traffic is small: token IDs, fused features, and metadata—**not** KV.