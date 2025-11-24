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

REVIEW

# EAGLE3 Implementation Deep Dive

## 1. Logical Control Flow

### Overview Architecture
```
Target Model (Base LLM) ←→ Draft Model (EAGLE3 Head)
     ↓                              ↓
  Hidden States              Draft Tokens
  (multi-layer)                   ↓
                            Tree Structure
                                  ↓
                         Parallel Verification
```

### A. Initialization (Prefill Phase)

**Step 0: Initial Forward Pass**
```
Input: prompt tokens [batch_size, prompt_len]

Target Model Forward:
├─ Run through all decoder layers
├─ Extract hidden states from 3 layers:
│  ├─ Low-level:  layer[2]           → l [batch_size, seq_len, hidden_size]
│  ├─ Mid-level:  layer[num_layers//2] → m [batch_size, seq_len, hidden_size]  
│  └─ High-level: layer[num_layers-3]  → h [batch_size, seq_len, hidden_size]
├─ Run through LM head
└─ Sample first token → t₁

Store: l, m, h feature sequences for reuse
```

**Step 1: Feature Fusion**
```
For each position i in sequence:
├─ Concatenate: [l[i], m[i], h[i]] → [batch_size, seq_len, 3*hidden_size]
├─ Project: fc(concat) → g[i] [batch_size, seq_len, hidden_size]
└─ g = fused multi-layer features
```

### B. Draft Generation (Core Loop)

**Draft Model Architecture Components:**
```
Input Processing:
├─ Target features: g[1:t] [batch, t, hidden_size]
├─ Token embeddings: e[t+1] [batch, 1, hidden_size]  (from last sampled token)
└─ Draft outputs: a[t+1:t+k] [batch, k, hidden_size] (from previous draft steps)

Draft Model Layers:
1. hidden_norm: RMSNorm on target hidden states
2. input_layernorm: RMSNorm on token embeddings
3. Concatenation: [e, hidden] → [batch, seq_len, 2*hidden_size]
4. LlamaDecoderLayeremb: Modified attention + MLP
   └─ First layer uses 2*hidden_size input (concat of embed + hidden)
   └─ Attention with custom tree mask
5. Output: a_next [batch, seq_len, hidden_size]
6. norm: RMSNorm on output
7. lm_head: Linear(hidden_size → draft_vocab_size)
8. Sample token: t_draft
```

**Draft Tree Generation (topK_genrate method):**

```python
# Shape notations
batch_size = 1  # typical
total_tokens = 63  # configurable draft budget
depth = 7  # tree depth
top_k = 10  # children per node

# Step 1: Generate root level drafts
input_ids = prefix_tokens [1, seq_len]
hidden_states = last_hidden_from_target [1, hidden_size]

# Forward through draft model
out_hidden, past_kv = draft_model(
    hidden_states,  # [1, seq_len, hidden_size]
    input_ids,      # [1, seq_len]
    use_cache=True
)

last_hidden = out_hidden[:, -1]  # [1, hidden_size]
logits = lm_head(norm(last_hidden))  # [1, draft_vocab_size]
log_probs = logsoftmax(logits)  # [1, draft_vocab_size]

# Sample top-k candidates
top_indices, top_scores = topk(log_probs, k=top_k)  # [1, top_k], [1, top_k]

# If vocab mapping exists (d2t buffer):
if draft_vocab_size < target_vocab_size:
    draft_token_ids = top_indices + d2t[top_indices]  # Map to target vocab
else:
    draft_token_ids = top_indices

# Store for tree building
scores_list = [top_scores]  # List of [1, top_k]
parents_list = [zeros(1)]   # Root has no parent
ss_token = [draft_token_ids]  # [1, top_k]

# Step 2-7: Expand tree level by level
input_ids = draft_token_ids  # [1, top_k]
input_hidden = last_hidden.repeat(1, top_k, 1)  # [1, top_k, hidden_size]
tree_mask = eye(top_k)[None, None]  # [1, 1, top_k, top_k]

for depth_i in range(depth):
    # Set attention mask for tree structure
    self.tree_mask = tree_mask
    
    # Position IDs increment by 1 at each level
    position_ids = len_posi + arange(top_k)  # [top_k]
    
    # Forward draft model
    out_hidden, past_kv = draft_model(
        input_hidden,     # [1, top_k, hidden_size]
        input_ids,        # [1, top_k]
        past_kv,
        position_ids,     # [top_k]
        use_cache=True
    )
    
    # Get logits for all positions
    logits = lm_head(norm(out_hidden[0]))  # [top_k, draft_vocab_size]
    log_probs = logsoftmax(logits)  # [top_k, draft_vocab_size]
    
    # Get top-k for each position
    topk_indices, topk_probs = topk(log_probs, k=top_k, dim=-1)  # [top_k, top_k]
    
    # Compute cumulative scores
    cu_scores = topk_probs + scores[:, None]  # [top_k, top_k]
    
    # Select best top_k across all (top_k × top_k) candidates
    topk_cs_indices, topk_cs_scores = topk(cu_scores.view(-1), k=top_k)  # [top_k]
    
    # Determine parent IDs
    out_ids = topk_cs_indices // top_k  # Which parent node [top_k]
    token_offset = topk_cs_indices % top_k  # Which child token [top_k]
    
    # Update for next level
    input_hidden = out_hidden[:, out_ids]  # [1, top_k, hidden_size]
    input_ids = topk_indices.view(-1)[topk_cs_indices][None]  # [1, top_k]
    
    if vocab_mapping:
        input_ids = input_ids + d2t[input_ids]
    
    scores = topk_cs_scores
    scores_list.append(cu_scores)
    
    # Build parent relationships
    bias = 1 + top_k**2 * max(0, depth_i-1) + (top_k if depth_i > 0 else 0)
    parents = topk_cs_index + bias
    parents_list.append(parents)
    
    # Update tree mask (allows attending to selected parents)
    tree_mask = cat([tree_mask[:,:,out_ids], eye(top_k)[None,None]], dim=3)
    
    ss_token.append(topk_indices)
    len_posi += 1

# Step 3: Select best total_tokens candidates
all_scores = cat(scores_list).view(-1)  # All generated scores
all_tokens = cat(ss_token).view(-1)     # All generated tokens
top_indices = topk(all_scores, k=total_tokens).indices
top_indices = sort(top_indices)

draft_tokens = all_tokens[top_indices]  # [total_tokens]
draft_tokens = cat([sample_token, draft_tokens])  # Prepend root token

# Step 4: Build verification tree structure
draft_parents = cat(parents_list)[top_indices // top_k]  # Parent pointers

# Create mask_index: maps each node to its parent position in selected set
mask_index = searchsorted(top_indices, draft_parents - 1)
mask_index[draft_parents == 0] = -1
mask_index = mask_index + 1

# Build tree attention mask
tree_mask = eye(total_tokens + 1).bool()
tree_mask[:, 0] = True  # All can attend to root

for i in range(total_tokens):
    tree_mask[i+1] += tree_mask[mask_index[i]]  # Inherit parent's mask

tree_position_ids = sum(tree_mask, dim=1) - 1  # Depth of each node

# Step 5: Build retrieve_indices for gathering accepted paths
noleaf_index = unique(mask_index)  # Intermediate nodes
leaf_num = total_tokens - len(noleaf_index) + 1

retrieve_indices = zeros(leaf_num, max_depth) - 1

rid = 0
for i in range(total_tokens + 1):
    if i not in noleaf_index:  # Is a leaf
        # Trace path from leaf to root
        cid = i
        depth = tree_position_ids[i]
        for j in reversed(range(depth + 1)):
            retrieve_indices[rid][j] = cid
            cid = mask_index[cid - 1]
        rid += 1

# Sort by path structure
retrieve_indices = sorted(retrieve_indices, key=custom_sort)
retrieve_indices = tensor(retrieve_indices)  # [leaf_num, max_depth]

return draft_tokens, retrieve_indices, tree_mask, tree_position_ids
```

**Shapes Summary for Draft Generation:**
```
draft_tokens:        [1, total_tokens+1]
retrieve_indices:    [num_leaves, max_depth]  # Paths from root to each leaf
tree_mask:          [1, 1, total_tokens+1, total_tokens+1]
tree_position_ids:  [total_tokens+1]  # Depth of each node in tree
```

### C. Verification Phase

**Step 1: Prepare Tree Inputs**
```python
# draft_tokens: [1, total_tokens+1] e.g., [1, 64]
# tree_position_ids: [64]

# Adjust positions based on current sequence length
position_ids = tree_position_ids + input_ids.shape[1]  # [64]

# Set tree mask on target model
target_model.model.tree_mask = tree_mask  # [1, 1, 64, 64]
```

**Step 2: Target Model Verification (tree_decoding)**
```python
# Forward target model with tree attention
outputs, logits, hidden_states = target_model(
    draft_tokens,      # [1, 64]
    past_key_values,
    position_ids,      # [64]
    use_cache=True
)

# Extract logits for leaf paths
logits = logits[0, retrieve_indices]  # [num_leaves, max_depth, vocab_size]

# For EAGLE3: extract multi-layer hidden states
hidden_states = cat([
    outputs.hidden_states[2],   # Low
    outputs.hidden_states[num_layers//2],  # Mid  
    outputs.hidden_states[num_layers-3]    # High
], dim=-1)  # [1, 64, 3*hidden_size]
```

**Step 3: Acceptance Evaluation (evaluate_posterior)**

*Greedy Case (temperature=0):*
```python
# draft_tokens shaped by retrieve_indices: [num_leaves, max_depth]
candidates = draft_tokens[0, retrieve_indices]  # [num_leaves, max_depth]

# Check if draft matches argmax of logits
posterior_mask = (candidates[:, 1:] == argmax(logits[:, :-1], dim=-1))  # [num_leaves, max_depth-1]

# Count consecutive matches per path
accept_lengths = cumprod(posterior_mask, dim=1).sum(dim=1)  # [num_leaves]
accept_length = accept_lengths.max()  # Longest accepted path

best_candidate = argmax(accept_lengths)  # Which leaf path
sample_p = logits[best_candidate, accept_length]  # Next token logits
```

*Sampling Case (temperature>0):*
```python
accept_length = 1
accept_cand = candidates[0][:1]
best_candidate = 0

for i in range(1, candidates.shape[1]):
    if i != accept_length:
        break
    
    # Find candidates with same prefix
    is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
    fi = nonzero(is_eq)[0][0]
    
    # Get ground truth distribution
    gt_logits = logits[fi, i-1]
    gt_logits = logits_processor(None, gt_logits)
    gtp = softmax(gt_logits, dim=0)
    
    # Try each unique candidate
    candidates_set = []
    for j in range(candidates.shape[0]):
        if is_eq[j]:
            x = candidates[j, i]
            if x in candidates_set or x == -1:
                continue
            candidates_set.append(x)
            
            # Accept with probability min(1, p(x)/q(x))
            r = random()
            px = gtp[x]
            qx = 1.0  # Draft prob (not tracked in EAGLE3)
            acp = px / qx
            
            if r <= acp:
                accept_cand = cat([accept_cand, x[None]])
                accept_length += 1
                best_candidate = j
                break
            else:
                # Adjust distribution
                gtp[x] = 0
                gtp = gtp / gtp.sum()

sample_p = gtp if adjusted else softmax(logits[best_candidate, accept_length-1])
```

**Shapes in Verification:**
```
candidates:      [num_leaves, max_depth]  # All possible draft paths
logits:          [num_leaves, max_depth, vocab_size]  # Target model predictions
accept_length:   scalar  # How many tokens accepted
best_candidate:  scalar  # Which leaf path was best
sample_p:        [vocab_size]  # Distribution for next token
```

### D. State Update (update_inference_inputs)

```python
prev_len = input_ids.shape[1]  # e.g., 100

# Select accepted tokens
select_indices = retrieve_indices[best_candidate, :accept_length+1] + prev_len
# e.g., [100, 103, 107] if accept_length=2

# Update KV cache by copying selected positions
for past_kv_data in past_key_values_data_list:
    # past_kv_data: [batch, num_heads, max_seq_len, head_dim]
    tgt = past_kv_data[..., select_indices, :]  # [batch, num_heads, accept_length+1, head_dim]
    dst = past_kv_data[..., prev_len:prev_len+tgt.shape[-2], :]
    dst.copy_(tgt)  # In-place update

# Update sequence length
current_length_data.fill_(prev_len + tgt.shape[-2])

# Append accepted tokens to input_ids
input_ids = cat([input_ids, candidates[best_candidate, :accept_length+1]], dim=-1)

# Sample next token
if logits_processor:
    next_token = multinomial(sample_p, 1)[None]
else:
    next_token = argmax(sample_p)[None, None]

# Extract accepted hidden states for next draft
retrieve_hidden = hidden_state_new[:, retrieve_indices]  # [1, num_leaves, max_depth, 3*hidden_size]
accept_hidden = retrieve_hidden[:, best_candidate, :accept_length+1]  # [1, accept_length+1, 3*hidden_size]

# Generate new draft tree
draft_tokens, retrieve_indices, tree_mask, tree_position_ids = draft_model.topK_genrate(
    accept_hidden,  # [1, accept_length+1, 3*hidden_size]
    cat([input_ids, next_token], dim=1),
    target_model.lm_head,
    logits_processor
)

return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, next_token
```

---

## 2. Key Architectural Differences from Standard Speculative Decoding

### A. Multi-Layer Feature Fusion (Core Innovation)

**Standard Spec Dec / EAGLE-2:**
```
Target Model → Top Layer Features → Draft Model
                [hidden_size]
```

**EAGLE3:**
```
Target Model → Low-layer features (layer 2)       → l [hidden_size]
            → Mid-layer features (layer N/2)    → m [hidden_size]
            → High-layer features (layer N-3)   → h [hidden_size]
            
Concatenate: [l, m, h] → [3*hidden_size]
            ↓
FC Layer: Linear(3*hidden_size → hidden_size, bias=False)
            ↓
Fused features: g [hidden_size]
```

**Implementation in Model Forward:**
```python
# From ea_model.py:forward()
outputs = target_model.model(
    input_ids,
    attention_mask,
    past_key_values,
    position_ids,
    output_hidden_states=True  # KEY: Must extract intermediate layers
)

# In modeling_llama_kv.py, layers store hidden states:
for idx, decoder_layer in enumerate(layers):
    if idx == len(layers)-3 or idx == len(layers)//2 or idx == 2:
        all_hidden_states += (hidden_states,)
    # ... layer forward ...

# In Eagle3LlamaForCausalLM:
def combine_hidden_states(self, hidden_states: Tensor) -> Tensor:
    # hidden_states: [batch, seq_len, 3*hidden_size]
    # Assumes hidden_states is already concatenated [l, m, h]
    return self.model.fc(hidden_states)  # → [batch, seq_len, hidden_size]
```

### B. Draft Model Architecture (LlamaDecoderLayeremb)

**Key Innovation: Concatenates Embeddings + Hidden States**

```python
class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        # MODIFIED: q/k/v_proj take 2*hidden_size input
        # self_attn.q_proj: Linear(2*hidden_size → num_heads * head_dim)
        # self_attn.k_proj: Linear(2*hidden_size → num_kv_heads * head_dim)
        # self_attn.v_proj: Linear(2*hidden_size → num_kv_heads * head_dim)
        
        self.mlp = LlamaMLP(config)
        
        # Normalization layers
        self.hidden_norm = LlamaRMSNorm(hidden_size)      # For target hidden states
        self.input_layernorm = LlamaRMSNorm(hidden_size)  # For token embeddings
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size)
    
    def forward(self, input_emb, hidden_states, ...):
        # input_emb: [batch, seq_len, hidden_size] - token embeddings
        # hidden_states: [batch, seq_len, hidden_size] - fused target features
        
        residual = hidden_states
        
        # Normalize separately
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        
        # KEY: Concatenate before attention
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        # → [batch, seq_len, 2*hidden_size]
        
        # Self-attention with 2*hidden_size input
        hidden_states, _, past_kv = self.self_attn(
            hidden_states,  # [batch, seq_len, 2*hidden_size]
            attention_mask,
            position_ids,
            past_key_value,
            use_cache=True
        )
        # → [batch, seq_len, hidden_size]
        
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return (hidden_states,)
```

**Full Draft Model (cnets.py):**
```python
class Model(nn.Module):
    def __init__(self, config, load_emb=False, path=None, 
                 total_tokens=63, depth=5, top_k=8):
        super().__init__()
        
        # Token embeddings (frozen, shared with target)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)
        if load_emb:
            # Load from target model
            self.embed_tokens.weight.data = load_target_embeddings()
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        
        # Single decoder layer (core of draft model)
        self.midlayer = LlamaDecoderLayeremb(config)
        
        # Feature fusion from target's multi-layer
        if hasattr(config, "target_hidden_size"):
            # For cross-model size (e.g., 70B target, 8B draft)
            self.fc = nn.Linear(target_hidden_size * 3, hidden_size, bias=False)
        else:
            self.fc = nn.Linear(hidden_size * 3, hidden_size, bias=False)
        
        # Output normalization and head
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, draft_vocab_size, bias=False)
        
        # Vocab mapping (optional, for compressed draft vocab)
        d2t = zeros(draft_vocab_size, dtype=long)  # Draft to target offset
        t2d = zeros(target_vocab_size, dtype=bool)  # Target in draft mask
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)
        
        # Hyperparameters
        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        
        # For tree generation
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, hidden_states, input_ids, 
                attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None):
        # hidden_states: [batch, seq_len, 3*hidden_size] OR [batch, seq_len, hidden_size]
        # input_ids: [batch, seq_len]
        
        batch_size, seq_length, _ = hidden_states.shape
        
        # Get token embeddings (frozen)
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
        
        # Fuse multi-layer features if needed
        if hidden_states.shape[-1] != inputs_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)  # 3*hidden_size → hidden_size
        
        # Prepare attention mask
        attention_mask = self._prepare_decoder_attention_mask(...)
        
        # Forward through single decoder layer
        out = self.midlayer(
            input_emb=inputs_embeds,     # [batch, seq_len, hidden_size]
            hidden_states=hidden_states,  # [batch, seq_len, hidden_size]
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values[0] if past_key_values else None,
            use_cache=True
        )
        
        hidden_states = out[0]
        
        if use_cache:
            return hidden_states, (out[2],)  # Return hidden + past_kv
        return hidden_states
```

### C. Vocabulary Mapping (d2t / t2d)

**Purpose:** Allow draft model to use smaller vocabulary than target

**d2t (Draft-to-Target) Buffer:**
```python
# Shape: [draft_vocab_size]
# Contains OFFSET to add to draft token ID to get target token ID

# Example: draft_vocab_size=32000, target_vocab_size=128000
# If draft token 1000 maps to target token 50000:
d2t[1000] = 49000  # offset

# During inference:
draft_id = topk(logits, k=top_k).indices  # e.g., [1000]
target_id = draft_id + d2t[draft_id]      # [1000 + 49000] = [50000]
```

**t2d (Target-to-Draft) Boolean Mask:**
```python
# Shape: [target_vocab_size]
# True if target token ID exists in draft vocabulary

# Example:
t2d[50000] = True   # This target token is in draft vocab
t2d[70000] = False  # This target token not in draft vocab
```

**Generation Process (preprocessing.py):**
```python
def generate_vocab_mapping_file(dataset, target_vocab_size, draft_vocab_size):
    # Count token frequencies in training data
    token_dict = Counter()
    for item in dataset:
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        masked_ids = input_ids[loss_mask == 1]  # Only count output tokens
        unique_ids, counts = masked_ids.unique(return_counts=True)
        token_dict.update(dict(zip(unique_ids, counts)))
    
    # Select top draft_vocab_size most frequent tokens
    top_N = token_dict.most_common(draft_vocab_size)
    used_tokens = sorted([key for key, freq in top_N])
    
    # Build mappings
    d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
    t2d = [i in used_tokens for i in range(target_vocab_size)]
    
    return torch.tensor(d2t), torch.tensor(t2d)
```

**Usage in Model:**
```python
# In topK_genrate:
if self.config.vocab_size == self.config.draft_vocab_size:
    # No mapping needed
    ss_token.append(topk_index)
    input_ids = topk_index
else:
    # Map draft IDs to target IDs
    ss_token.append(topk_index + self.d2t[topk_index])
    input_ids = topk_index + self.d2t[topk_index]
```

### D. Training-Time Test (Key Innovation)

**Problem:** Draft model generates `a` vectors, not target features `g`
- Step 1 input: [g₁, g₂, ..., gₜ] (target features)
- Step 2+ input: [g₁, ..., gₜ, a_{t+1}, a_{t+2}] (mix of target + draft)

**Solution:** Simulate multi-step during training

**Training Loop:**
```python
# Step 1: Normal forward (input = target features)
a_{t+1} = draft_model(g[1:t], e[t])  # e = token embedding
loss_1 = CrossEntropy(lm_head(norm(a_{t+1})), target_token_{t+1})

# Step 2: Training-time test (input includes draft output a)
a_{t+2} = draft_model(
    torch.cat([g[1:t], a_{t+1}], dim=1),  # Mix target + draft
    e[t+1]
)
loss_2 = CrossEntropy(lm_head(norm(a_{t+2})), target_token_{t+2})

# Step 3+: Continue pattern
# ...

total_loss = loss_1 + loss_2 + loss_3 + ...
```

**Modified Attention for Training (cnets.py):**
```python
def _prepare_decoder_attention_mask(...):
    # Standard causal mask
    combined_attention_mask = _make_causal_mask(...)
    
    # ADD TREE MASK for training-time test
    if hasattr(self, "tree_mask") and self.tree_mask is not None:
        tree_mask = self.tree_mask
        _, _, tree_h, tree_w = tree_mask.shape
        
        # Apply tree mask to bottom-right corner
        combined_attention_mask[:, :, -tree_h:, -tree_w:][tree_mask == 0] = \
            torch.finfo(float32).min
    
    return combined_attention_mask
```

**Tree Mask Pattern (from paper Figure 6):**
```
Original sequence:    [How, can, I]
Predictions Step 1:   [are, we, do]  (predict from "How can I")
Predictions Step 2:   [you, help, it] (predict from "How can I are/we/do")

Attention Mask:
     How can I are we do you help it
How  ✓
can  ✓   ✓
I    ✓   ✓   ✓
are  ✓   ✓   ✓   ✓              <-- can attend to "How can I" + self
we   ✓   ✓   ✓       ✓          <-- can attend to "How can I" + self
do   ✓   ✓   ✓           ✓      <-- can attend to "How can I" + self
you  ✓   ✓   ✓   ✓           ✓  <-- can attend to "How can I are" + self
help ✓   ✓   ✓       ✓           ✓  <-- can attend to "How can I we" + self
it   ✓   ✓   ✓           ✓           ✓  <-- can attend to "How can I do" + self
```

### E. No Feature Prediction Loss

**EAGLE/EAGLE-2:**
```python
loss = loss_feature + loss_token
# where loss_feature = MSE(predicted_feature, target_feature)
```

**EAGLE3:**
```python
loss = loss_token  # ONLY token prediction
# No constraint that draft output must match target features
```

This allows:
1. Draft model full expressiveness (not constrained to mimic target features)
2. Different input types (fused features instead of top-layer only)
3. Better scaling with more training data

---

## 3. Complete Shape Reference

```python
# Global Constants
batch_size = 1                    # Typically 1 for inference
hidden_size = 4096               # e.g., for LLaMA-8B
draft_vocab_size = 32000         # Draft model vocab (can be < target)
target_vocab_size = 128000       # Target model vocab
num_layers = 32                  # Target model layers
total_tokens = 63                # Draft budget
depth = 7                        # Tree depth
top_k = 10                       # Children per node
max_depth = 8                    # Max path length in tree
num_leaves = ~40                 # Leaf nodes in tree
seq_len = 100                    # Example current sequence length

# Target Model Outputs
outputs.hidden_states[2]:        [1, seq_len, hidden_size]      # Low layer
outputs.hidden_states[16]:       [1, seq_len, hidden_size]      # Mid layer  
outputs.hidden_states[29]:       [1, seq_len, hidden_size]      # High layer
concatenated:                    [1, seq_len, 3*hidden_size]

# Draft Model Tensors
fused_features (g):              [1, seq_len, hidden_size]      # After fc
token_embeddings (e):            [1, 1, hidden_size]            # Current token
concat_input:                    [1, seq_len, 2*hidden_size]    # For attention
draft_output (a):                [1, seq_len, hidden_size]      # Draft features
logits:                          [1, seq_len, draft_vocab_size] # Before sampling

# Tree Structure
draft_tokens:                    [1, total_tokens+1]            # e.g., [1, 64]
tree_mask:                       [1, 1, 64, 64]                 # Attention mask
tree_position_ids:               [64]                           # Depth per node
retrieve_indices:                [num_leaves, max_depth]        # e.g., [40, 8]
parents_list:                    [64]                           # Parent pointers

# Verification
candidates:                      [num_leaves, max_depth]        # Draft paths
target_logits:                   [num_leaves, max_depth, target_vocab_size]
posterior_mask:                  [num_leaves, max_depth-1]      # Acceptance mask
accept_length:                   scalar                         # Accepted tokens

# KV Cache
past_key_values[i][0]:          [1, num_heads, max_seq_len, head_dim]  # Keys
past_key_values[i][1]:          [1, num_heads, max_seq_len, head_dim]  # Values

# Vocab Mapping
d2t:                            [draft_vocab_size]              # Draft→Target offset
t2d:                            [target_vocab_size]             # Bool mask
```

This comprehensive guide should enable faithful reimplementation of EAGLE3 with all critical architectural details and control flow.