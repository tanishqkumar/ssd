import torch
import torch.nn as nn
import numpy as np
import os
from transformers import LlamaConfig, AutoConfig
from ssd.models.ref import Model as RefModel, EagleConfig
from ssd.models.eagle3_draft_llama3 import Eagle3DraftForCausalLM
from ssd.utils.context import set_context, reset_context
from ssd.utils.loader import default_weight_loader

# --- Configuration & Helpers ---

class ModelComparator:
    def __init__(self, hidden_size=None, num_heads=None, num_kv_heads=None, seq_len=16, bs=2, device="cuda", dtype=torch.bfloat16, model_path=None, target_path=None):
        self.seq_len = seq_len
        self.bs = bs
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        self.target_path = target_path
        
        # Infer config from model_path if provided, similar to Config.__post_init__
        if model_path and os.path.isdir(model_path):
            print(f"Inferring config from {model_path}...")
            draft_hf_config = AutoConfig.from_pretrained(model_path)
            self.hidden_size = hidden_size or draft_hf_config.hidden_size
            self.num_heads = num_heads or draft_hf_config.num_attention_heads
            self.num_kv_heads = num_kv_heads or draft_hf_config.num_key_value_heads
            self.intermediate_size = draft_hf_config.intermediate_size
            self.vocab_size = draft_hf_config.vocab_size
            self.max_position_embeddings = draft_hf_config.max_position_embeddings
            self.rms_norm_eps = draft_hf_config.rms_norm_eps
            self.rope_theta = getattr(draft_hf_config, 'rope_theta', 10000.0)
            print(f"Inferred: hidden_size={self.hidden_size}, num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}")
        else:
            # Use defaults for testing
            self.hidden_size = hidden_size or 256
            self.num_heads = num_heads or 4
            self.num_kv_heads = num_kv_heads or 2
            self.intermediate_size = self.hidden_size * 4
            self.vocab_size = 1000
            self.max_position_embeddings = 1024
            self.rms_norm_eps = 1e-5
            self.rope_theta = 10000.0
        
        # For eagle3, target hidden size is inferred from target_path or assumed same
        if target_path and os.path.isdir(target_path):
            target_hf_config = AutoConfig.from_pretrained(target_path)
            self.target_hidden_size = target_hf_config.hidden_size
        else:
            self.target_hidden_size = self.hidden_size
        
        self.setup_models()
        
    def setup_models(self):
        print(f"Setting up models on {self.device} with {self.dtype}...")
        
        # If loading from path, read config first to get correct vocab sizes
        if self.model_path:
            bin_file = os.path.join(self.model_path, "pytorch_model.bin")
            if os.path.exists(bin_file):
                state_dict = torch.load(bin_file, map_location="cpu")
                if 'd2t' in state_dict and 't2d' in state_dict:
                    target_vocab_size = len(state_dict['t2d'])  # target vocab size
                    draft_vocab_size = len(state_dict['d2t'])  # draft vocab size
                    print(f"Detected target_vocab_size={target_vocab_size}, draft_vocab_size={draft_vocab_size} from weights")
                    # Update vocab_size if different from what was loaded
                    if target_vocab_size != self.vocab_size:
                        print(f"Warning: target_vocab_size from weights ({target_vocab_size}) differs from config ({self.vocab_size}), using weights")
                        self.vocab_size = target_vocab_size
                else:
                    draft_vocab_size = self.vocab_size
            else:
                draft_vocab_size = self.vocab_size
        else:
            draft_vocab_size = self.vocab_size
        
        # Infer eagle_layers from target config like Config does
        if self.target_path and os.path.isdir(self.target_path):
            target_hf_config = AutoConfig.from_pretrained(self.target_path)
            L = target_hf_config.num_hidden_layers
            eagle_layers = [2, L//2, L-3]
            print(f"Inferred eagle_layers={eagle_layers} from target model with {L} layers")
        else:
            eagle_layers = [2, 16, 29]  # Default for Llama 3.1 8B
        
        # 1. Ref Config
        ref_config = EagleConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            vocab_size=self.vocab_size,
            draft_vocab_size=draft_vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            target_hidden_size=self.target_hidden_size,
            pad_token_id=0,
            hidden_act="silu"
        )
        self.ref_model = RefModel(ref_config).to(device=self.device, dtype=self.dtype)
        self.ref_model.eval()
        
        # 2. Our Config
        our_config = LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=1, 
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            hidden_act="silu",
            tie_word_embeddings=False,
        )
        our_config.draft_vocab_size = draft_vocab_size
        
        self.our_model = Eagle3DraftForCausalLM(
            config=our_config,
            draft=True,
            use_eagle=True,
            eagle_layers=eagle_layers,
            d_model_target=self.target_hidden_size,
            spec_k=1
        ).to(device=self.device, dtype=self.dtype)
        self.our_model.eval()
        
        if self.model_path:
            self.load_weights_from_disk()
        else:
            self.copy_weights()
        self.allocate_kv_cache()
        if not self.model_path:
            self.setup_vocab_mappings()
    
    def allocate_kv_cache(self):
        """Allocate KV cache for the our_model's attention layers."""
        # Simple allocation for testing - just enough for our test sequences
        block_size = 16
        num_blocks = 64  # Enough for our test
        num_kv_heads = self.num_kv_heads
        head_dim = self.hidden_size // self.num_heads
        
        # Allocate cache [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        # For our single-layer model
        kv_cache = torch.zeros(
            2, 1, num_blocks, block_size, num_kv_heads, head_dim,
            dtype=self.dtype, device=self.device
        )
        
        # Assign to attention modules
        for module in self.our_model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, 0]  # [num_blocks, block_size, num_kv_heads, head_dim]
                module.v_cache = kv_cache[1, 0]
                print(f"Allocated KV cache with shape: {module.k_cache.shape}")
    
    def setup_vocab_mappings(self):
        """Setup d2t and t2d vocab mappings for Eagle3 model."""
        # For testing, use identity mapping since draft_vocab_size == vocab_size
        draft_vocab_size = self.our_model.config.draft_vocab_size
        self.our_model.d2t_tensor = torch.zeros(draft_vocab_size, dtype=torch.int64, device=self.device)
        self.our_model.t2d_tensor = torch.zeros(self.vocab_size, dtype=torch.int64, device=self.device)
        print(f"Setup vocab mappings (identity) for vocab_size={self.vocab_size}, draft_vocab_size={draft_vocab_size}")
    
    def load_weights_from_disk(self):
        """Load weights from disk into both ref and our models (matching loader.py logic)."""
        print(f"[load_model] Detected EAGLE3 draft model, loading from pytorch_model.bin")
        bin_file = os.path.join(self.model_path, "pytorch_model.bin")
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"Expected pytorch_model.bin at {bin_file}")
        
        state_dict = torch.load(bin_file, map_location="cpu")
        
        # Load d2t and t2d for our model (matching loader.py lines 67-78)
        if hasattr(self.our_model, 'd2t') and 'd2t' in state_dict:
            d2t_tensor = state_dict['d2t']
            self.our_model.d2t = {i: int(d2t_tensor[i].item()) for i in range(len(d2t_tensor))}
            self.our_model.d2t_tensor = d2t_tensor.to(self.device).long()
            print(f"[load_model] Loaded d2t dictionary with {len(self.our_model.d2t)} entries")
        
        if hasattr(self.our_model, 't2d') and 't2d' in state_dict:
            t2d_tensor = state_dict['t2d']
            self.our_model.t2d = {i: int(t2d_tensor[i].item()) for i in range(len(t2d_tensor))}
            self.our_model.t2d_tensor = t2d_tensor.to(self.device).long()
            print(f"[load_model] Loaded t2d dictionary with {len(self.our_model.t2d)} entries")
        
        # Check for embedding layer (matching loader.py lines 80-106)
        found_embed_tokens = False
        found_any_embed = False
        for weight_name in state_dict.keys():
            if 'embed' in weight_name.lower():
                found_any_embed = True
                if 'embed_tokens' in weight_name:
                    found_embed_tokens = True
                    break
        
        if not found_embed_tokens:
            if self.target_path:
                print(f"[load_model] 'embed_tokens' not found in draft weights. Attempting to load from target path: {self.target_path}")
                if self.load_embeddings_from_target():
                    found_embed_tokens = True
            
            if not found_embed_tokens:
                if found_any_embed:
                    raise ValueError(
                        f"[load_model] ERROR: Found embedding layer(s) in EAGLE3 weights but not 'embed_tokens'. "
                        f"Available weights: {list(state_dict.keys())}"
                    )
                else:
                    raise ValueError(
                        f"[load_model] ERROR: No embedding layer found in EAGLE3 weights. "
                        f"Expected 'embed_tokens' or similar. Available weights: {list(state_dict.keys())}"
                    )
        
        # Get packed modules mapping for our model
        packed_modules_mapping = self.our_model.packed_modules_mapping
        
        # Load model weights (matching loader.py lines 108-146)
        print(f"[load_model] Loading EAGLE3 weights...")
        for weight_name, loaded_weight in state_dict.items():
            # Skip dictionary tensors
            if weight_name in ['d2t', 't2d']:
                continue
            
            # Load into our model first (using weight_loader like loader.py)
            # Check if this weight should use packed module loading
            is_packed = False
            for k, (v, shard_id) in packed_modules_mapping.items():
                if k in weight_name:
                    param_name = weight_name.replace(k, v)
                    param = self.our_model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, loaded_weight, shard_id)
                    is_packed = True
                    break
            
            if not is_packed:
                # Map EAGLE3 weight names to our architecture for unpacked weights
                if weight_name == 'midlayer.hidden_norm.weight':
                    param_name = 'model.layer.conditioning_feature_ln.weight'
                elif weight_name.startswith('midlayer.'):
                    param_name = weight_name.replace('midlayer.', 'model.layer.')
                elif weight_name == 'norm.weight':
                    param_name = 'final_norm.weight'
                else:
                    # fc.weight, lm_head.weight, embed_tokens.weight stay the same
                    param_name = weight_name
                
                try:
                    param = self.our_model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                except Exception as e:
                    print(f"Warning: Could not load {weight_name} -> {param_name} into our model: {e}")
            
            # Also load into ref model (convert to device/dtype manually)
            loaded_weight_device = loaded_weight.to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                try:
                    if weight_name == 'midlayer.hidden_norm.weight':
                        self.ref_model.midlayer.hidden_norm.weight.data.copy_(loaded_weight_device)
                    elif weight_name.startswith('midlayer.'):
                        param_name = weight_name.replace('midlayer.', '')
                        parts = param_name.split('.')
                        obj = self.ref_model.midlayer
                        for part in parts[:-1]:
                            obj = getattr(obj, part)
                        getattr(obj, parts[-1]).data.copy_(loaded_weight_device)
                    elif weight_name == 'fc.weight':
                        self.ref_model.fc.weight.data.copy_(loaded_weight_device)
                    elif weight_name == 'norm.weight':
                        self.ref_model.norm.weight.data.copy_(loaded_weight_device)
                    elif weight_name == 'lm_head.weight':
                        self.ref_model.lm_head.weight.data.copy_(loaded_weight_device)
                    elif weight_name == 'embed_tokens.weight':
                        self.ref_model.embed_tokens.weight.data.copy_(loaded_weight_device)
                except Exception as e:
                    pass  # Silently skip ref model errors
        
        print("[load_model] Finished loading weights from disk")
    
    def load_embeddings_from_target(self):
        """Load embedding weights from target model (exact same logic as loader.py)"""
        if not self.target_path:
            print("Warning: No target_path provided, cannot load embeddings")
            return False
        
        target_keys = ["model.embed_tokens.weight", "embed_tokens.weight"]
        
        # Try safetensors first
        import glob
        safetensor_files = glob.glob(os.path.join(self.target_path, "*.safetensors"))
        for file in safetensor_files:
            try:
                from safetensors import safe_open
                with safe_open(file, "pt", "cpu") as f:
                    keys = f.keys()
                    for key in target_keys:
                        if key in keys:
                            print(f"[load_model] Found embedding {key} in {file}")
                            tensor = f.get_tensor(key)
                            # Load into our_model using weight_loader (like loader.py)
                            param = self.our_model.get_parameter("model.embed_tokens.weight")
                            weight_loader = getattr(param, "weight_loader", default_weight_loader)
                            weight_loader(param, tensor)
                            # Also load into ref_model
                            tensor_device = tensor.to(device=self.device, dtype=self.dtype)
                            with torch.no_grad():
                                self.ref_model.embed_tokens.weight.data.copy_(tensor_device)
                            print(f"[load_model] Loaded embeddings from target model")
                            return True
            except Exception as e:
                print(f"[load_model] Error reading safetensor {file}: {e}")
                continue
        
        # Try bin files
        bin_files = glob.glob(os.path.join(self.target_path, "pytorch_model*.bin"))
        for file in bin_files:
            try:
                print(f"[load_model] Checking {file} for embeddings...")
                state_dict = torch.load(file, map_location="cpu")
                for key in target_keys:
                    if key in state_dict:
                        print(f"[load_model] Found embedding {key} in {file}")
                        tensor = state_dict[key]
                        # Load into our_model using weight_loader (like loader.py)
                        param = self.our_model.get_parameter("model.embed_tokens.weight")
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, tensor)
                        # Also load into ref_model
                        tensor_device = tensor.to(device=self.device, dtype=self.dtype)
                        with torch.no_grad():
                            self.ref_model.embed_tokens.weight.data.copy_(tensor_device)
                        print(f"[load_model] Loaded embeddings from target model")
                        return True
            except Exception as e:
                print(f"[load_model] Error reading bin {file}: {e}")
                continue
        
        print("Warning: Could not load embeddings from target model")
        return False
        
    def copy_weights(self):
        print("Copying weights...")
        with torch.no_grad():
            # FC
            self.our_model.fc.weight.copy_(self.ref_model.fc.weight)
            # Embed
            self.our_model.model.embed_tokens.weight.copy_(self.ref_model.embed_tokens.weight)
            # Head
            self.our_model.lm_head.weight.copy_(self.ref_model.lm_head.weight)
            # Final Norm
            self.our_model.final_norm.weight.copy_(self.ref_model.norm.weight)
            
            # Layer
            ref_layer = self.ref_model.midlayer
            our_layer = self.our_model.model.layer
            
            our_layer.input_layernorm.weight.copy_(ref_layer.input_layernorm.weight)
            our_layer.conditioning_feature_ln.weight.copy_(ref_layer.hidden_norm.weight)
            our_layer.post_attention_layernorm.weight.copy_(ref_layer.post_attention_layernorm.weight)
            
            # MLP
            gate = ref_layer.mlp.gate_proj.weight
            up = ref_layer.mlp.up_proj.weight
            our_layer.mlp.gate_up_proj.weight.copy_(torch.cat([gate, up], dim=0))
            our_layer.mlp.down_proj.weight.copy_(ref_layer.mlp.down_proj.weight)
            
            # Attn
            q = ref_layer.self_attn.q_proj.weight
            k = ref_layer.self_attn.k_proj.weight
            v = ref_layer.self_attn.v_proj.weight
            our_layer.self_attn.qkv_proj.weight.copy_(torch.cat([q, k, v], dim=0))
            our_layer.self_attn.o_proj.weight.copy_(ref_layer.self_attn.o_proj.weight)

    def check(self, name, ref, our, tol=1e-2):
        # Ensure shapes match before compare
        if ref.shape != our.shape:
            # flatten both
            ref = ref.view(-1)
            our = our.view(-1)
            
        diff = (ref - our).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        num_disagree = (diff >= 1e-5).sum().item() 
        total_entries = diff.numel()
        status = "MATCH" if max_diff < tol else "MISMATCH"
        print(f"[{status}] {name:<25}: max diff = {max_diff:.8f}, mean diff = {mean_diff:.8f}, disagree = {num_disagree}/{total_entries}")
        if max_diff >= tol:
            print(f"    Ref mean: {ref.float().mean():.4f}, std: {ref.float().std():.4f}")
            print(f"    Our mean: {our.float().mean():.4f}, std: {our.float().std():.4f}")
        return max_diff < tol

    def run_comparison(self):
        print("\n--- Starting Modular Comparison ---")
        torch.manual_seed(42)
        
        # 1. Prepare Inputs
        # Input IDs: [BS, SeqLen] for Ref, [BS*SeqLen] for Our
        input_ids_2d = torch.randint(0, self.vocab_size, (self.bs, self.seq_len), device=self.device)
        input_ids_1d = input_ids_2d.flatten()
        
        # Hidden States: [BS, SeqLen, D] for Ref, [BS*SeqLen, D] for Our
        hidden_states_2d = torch.randn(self.bs, self.seq_len, self.target_hidden_size * 3, device=self.device, dtype=self.dtype)
        hidden_states_1d = hidden_states_2d.view(-1, self.target_hidden_size * 3)
        
        # Positions: [BS, SeqLen] for Ref, [BS*SeqLen] for Our
        positions_2d = torch.arange(self.seq_len, device=self.device).unsqueeze(0).repeat(self.bs, 1)
        positions_1d = positions_2d.flatten().long()
        
        # 2. Setup Context for Our Model (FlashAttn)
        # cu_seqlens must be int32
        cu_seqlens = torch.arange(0, (self.bs + 1) * self.seq_len, self.seq_len, dtype=torch.int32, device=self.device)
        
        # Create slot_mapping: for prefill, maps each token position to its KV cache slot
        # Assuming block_size=16, we map positions sequentially
        block_size = 16
        total_tokens = self.bs * self.seq_len
        slot_mapping = torch.arange(total_tokens, dtype=torch.int32, device=self.device)
        
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=self.seq_len,
            max_seqlen_k=self.seq_len,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=None
        )
        
        with torch.no_grad():
            # 3. Embeddings & FC
            ref_emb = self.ref_model.embed_tokens(input_ids_2d) # [B, S, D]
            our_emb = self.our_model.model.embed_tokens(input_ids_1d) # [T, D]
            self.check("Embeddings", ref_emb, our_emb.view(self.bs, self.seq_len, -1))
            
            ref_fc = self.ref_model.fc(hidden_states_2d)
            our_fc = self.our_model.fc(hidden_states_1d)
            self.check("FC Projection", ref_fc, our_fc.view(self.bs, self.seq_len, -1))
            
            # 4. Pre-Norms
            ref_mid = self.ref_model.midlayer
            our_mid = self.our_model.model.layer
            
            ref_norm_h = ref_mid.hidden_norm(ref_fc)
            our_norm_h = our_mid.conditioning_feature_ln(our_fc)
            self.check("Conditioning LN", ref_norm_h, our_norm_h.view(self.bs, self.seq_len, -1))
            
            ref_norm_e = ref_mid.input_layernorm(ref_emb)
            our_norm_e = our_mid.input_layernorm(our_emb)
            self.check("Input LN", ref_norm_e, our_norm_e.view(self.bs, self.seq_len, -1))
            
            # 5. Cat
            ref_cat = torch.cat((ref_norm_e, ref_norm_h), dim=-1)
            our_cat = torch.cat([our_norm_e, our_norm_h], dim=-1) # [T, 2D]
            self.check("Attn Input (Cat)", ref_cat, our_cat.view(self.bs, self.seq_len, -1))
            
            # 6. Attention
            # Ref Mask
            ref_mask = self.ref_model._prepare_decoder_attention_mask(None, (self.bs, self.seq_len), ref_cat, 0)
            ref_attn_out, _, _ = ref_mid.self_attn(ref_cat, attention_mask=ref_mask, position_ids=positions_2d.long())
            
            # Our Attn (use 1D inputs directly)
            # our_cat is [T, 2D], positions_1d is [T]
            our_attn_out = our_mid.self_attn(positions_1d, our_cat)
            # Output is [T, D]
            
            self.check("Attention Output", ref_attn_out, our_attn_out.view(self.bs, self.seq_len, -1))
            
            # 7. Residuals & Post Norm
            ref_resid1 = ref_fc + ref_attn_out
            our_resid1 = our_fc + our_attn_out # [T, D]
            self.check("Residual 1", ref_resid1, our_resid1.view(self.bs, self.seq_len, -1))
            
            ref_pn = ref_mid.post_attention_layernorm(ref_resid1)
            our_pn, _ = our_mid.post_attention_layernorm(our_attn_out, our_fc)
            self.check("Post-Attn LN", ref_pn, our_pn.view(self.bs, self.seq_len, -1))
            
            # 8. MLP
            ref_mlp = ref_mid.mlp(ref_pn)
            our_mlp = our_mid.mlp(our_pn)
            self.check("MLP Output", ref_mlp, our_mlp.view(self.bs, self.seq_len, -1))
            
            # 9. Final Layer Out
            ref_out = ref_resid1 + ref_mlp
            our_out = our_resid1 + our_mlp
            self.check("Final Layer Output", ref_out, our_out.view(self.bs, self.seq_len, -1))
            
            # 10. Full Forward
            print("\n--- Full Forward ---")
            ref_final = self.ref_model(hidden_states_2d, input_ids_2d, position_ids=positions_2d.long(), use_cache=False)
            our_final = self.our_model(input_ids_1d, positions_1d, hidden_states_1d)
            self.check("Full Hidden States", ref_final, our_final.view(self.bs, self.seq_len, -1))
            
            # 11. Logits
            # For logits comparison, we need all token logits, not just last token
            # Temporarily set context to non-prefill so lm_head doesn't slice
            
            # Print stats before lm_head
            print(f"\nBefore lm_head:")
            print(f"  ref_final: mean={ref_final.mean().item():.6f}, std={ref_final.std().item():.6f}")
            print(f"  our_final: mean={our_final.mean().item():.6f}, std={our_final.std().item():.6f}")
            
            ref_normed = self.ref_model.norm(ref_final)
            our_normed = self.our_model.final_norm(our_final)
            
            print(f"\nAfter norm, before lm_head:")
            print(f"  ref_normed: mean={ref_normed.mean().item():.6f}, std={ref_normed.std().item():.6f}")
            print(f"  our_normed: mean={our_normed.mean().item():.6f}, std={our_normed.std().item():.6f}")
            
            ref_logits = self.ref_model.lm_head(ref_normed)  # [bs, seq_len, draft_vocab_size]
            
            # Compute our logits by directly calling the layers (bypass the slicing in lm_head)
            import torch.nn.functional as F
            our_logits_raw = F.linear(our_normed, self.our_model.lm_head.weight)  # [bs*seq_len, draft_vocab_size]
            
            print(f"\nAfter lm_head (before d2t expansion):")
            print(f"  ref_logits shape: {ref_logits.shape}, mean={ref_logits.mean().item():.6f}, std={ref_logits.std().item():.6f}")
            print(f"  our_logits_raw shape: {our_logits_raw.shape}, mean={our_logits_raw.mean().item():.6f}, std={our_logits_raw.std().item():.6f}")
            
            # Compare raw logits before d2t expansion (both should be draft_vocab_size)
            self.check("Final Logits (draft vocab)", ref_logits, our_logits_raw.view(self.bs, self.seq_len, -1))
            
            # Also show what d2t expansion would look like (for reference)
            B_total = our_logits_raw.shape[0]  # Should be bs * seq_len
            vocab_size = self.vocab_size
            draft_vocab_size = our_logits_raw.shape[-1]
            base = torch.arange(draft_vocab_size, device=our_logits_raw.device)
            target_indices = base + self.our_model.d2t_tensor
            our_logits_expanded = our_logits_raw.new_full((B_total, vocab_size), float('-inf'))
            our_logits_expanded[:, target_indices] = our_logits_raw
            
            print(f"\nAfter d2t expansion (for reference):")
            print(f"  our_logits_expanded shape: {our_logits_expanded.shape}")
            print(f"  Non-inf values: mean={our_logits_expanded[our_logits_expanded != float('-inf')].mean().item():.6f}, std={our_logits_expanded[our_logits_expanded != float('-inf')].std().item():.6f}")
            
            # Cleanup
            reset_context()

if __name__ == "__main__":
    import argparse
    
    # Default paths - using real EAGLE3 weights by default
    DEFAULT_TARGET_PATH = "/data/tkumar/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    # Try to find EAGLE3 weights - actual paths from your system
    POSSIBLE_DRAFT_PATHS = [
        "/data/tkumar/huggingface/hub/models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B",
        "/data/tkumar/huggingface/hub/models--yuhuili--EAGLE3-LLaMA3.3-Instruct-70B",
        "/data/tkumar/huggingface/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3",
    ]
    
    DEFAULT_DRAFT_PATH = None
    for path in POSSIBLE_DRAFT_PATHS:
        if os.path.isdir(path):
            # Check if it has snapshots subdirectory (HuggingFace hub structure)
            snapshots_dir = os.path.join(path, "snapshots")
            if os.path.isdir(snapshots_dir):
                # Get the first (usually only) snapshot
                snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshots:
                    DEFAULT_DRAFT_PATH = os.path.join(snapshots_dir, snapshots[0])
                    print(f"Found EAGLE3 draft model at: {DEFAULT_DRAFT_PATH}")
                    break
            else:
                # Direct path without snapshots
                DEFAULT_DRAFT_PATH = path
                print(f"Found EAGLE3 draft model at: {DEFAULT_DRAFT_PATH}")
                break
    
    if DEFAULT_DRAFT_PATH is None:
        print("Warning: No EAGLE3 draft model found in common locations. Searching for any pytorch_model.bin in data dir...")
        # Try to find any directory with pytorch_model.bin that might be EAGLE3
        import glob
        bin_files = glob.glob("/data/tkumar/**/pytorch_model.bin", recursive=True)
        for bin_file in bin_files:
            dir_path = os.path.dirname(bin_file)
            # Check if it has d2t/t2d (EAGLE3 signature)
            try:
                state_dict = torch.load(bin_file, map_location="cpu")
                if 'd2t' in state_dict and 't2d' in state_dict:
                    DEFAULT_DRAFT_PATH = dir_path
                    print(f"Found EAGLE3 draft model at: {DEFAULT_DRAFT_PATH}")
                    break
            except:
                continue
    
    parser = argparse.ArgumentParser(description="Compare EAGLE3 draft implementations")
    parser.add_argument("--model-path", type=str, default=DEFAULT_DRAFT_PATH, 
                        help=f"Path to EAGLE3 draft model (default: {DEFAULT_DRAFT_PATH or 'None - will use random weights'})")
    parser.add_argument("--target-path", type=str, default=DEFAULT_TARGET_PATH, 
                        help="Path to target model (for config inference and embeddings)")
    parser.add_argument("--hidden-size", type=int, default=None, 
                        help="Hidden size (auto-inferred from model-path if not provided)")
    parser.add_argument("--num-heads", type=int, default=None, 
                        help="Number of attention heads (auto-inferred from model-path if not provided)")
    parser.add_argument("--num-kv-heads", type=int, default=None, 
                        help="Number of KV heads (auto-inferred from model-path if not provided)")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length for test")
    parser.add_argument("--bs", type=int, default=2, help="Batch size for test")
    args = parser.parse_args()
    
    # Check if paths exist
    if args.model_path and not os.path.isdir(args.model_path):
        print(f"Warning: model-path {args.model_path} does not exist or is not a directory")
        args.model_path = None
    
    if args.target_path and not os.path.isdir(args.target_path):
        print(f"Warning: target-path {args.target_path} does not exist or is not a directory")
        args.target_path = None
    
    if args.model_path:
        print(f"Loading EAGLE3 draft model from: {args.model_path}")
    else:
        print("No EAGLE3 draft model found - using random weights for comparison")
    
    comp = ModelComparator(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        seq_len=args.seq_len,
        bs=args.bs,
        model_path=args.model_path,
        target_path=args.target_path
    )
    comp.run_comparison()
