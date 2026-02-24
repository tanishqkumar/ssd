import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig  # Changed from Qwen3Config

from ssd.layers.activation import SiluAndMul
from ssd.layers.attention import Attention
from ssd.layers.layernorm import RMSDNorm
from ssd.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from ssd.layers.rotary_embedding import get_rope
from ssd.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from ssd.models.llama3 import LlamaAttention, LlamaMLP


class Eagle3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        rms_norm_eps: float,
        head_dim: int | None,
        rope_theta: float,
        rope_scaling: dict | None,
        draft: bool,
        speculate: bool,
        spec_k: int,
        async_fan_out: int,
        draft_async: bool,
        tp_group: dist.ProcessGroup | None,
        tp_size: int,
    ):
        super().__init__()
        self.draft = draft
        self.draft_async = draft_async
        self.tp_group = tp_group
        self.tp_size = tp_size
        
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        
        self.qkv_proj = QKVParallelLinear(
            2 * hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            tp_group=self.tp_group,
            tp_size=self.tp_size,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            tp_group=self.tp_group,
            tp_size=self.tp_size,
        )
        
        if rope_scaling is not None:
            rope_scaling = None
        
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            draft=draft,
            speculate=speculate,
            draft_async=draft_async,
            use_eagle=True,
            F=async_fan_out,
            K=spec_k,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Ensure all inputs to attn are contiguous
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Eagle3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        draft: bool,
        speculate: bool,
        spec_k: int,
        async_fan_out: int,
        draft_async: bool,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__()
        self.self_attn = Eagle3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 500000),
            rope_scaling=getattr(config, "rope_scaling", None),
            draft=draft,
            speculate=speculate,
            spec_k=spec_k,
            async_fan_out=async_fan_out,
            draft_async=draft_async,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.input_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.conditioning_feature_ln = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        token_embeddings: torch.Tensor,
        conditioning_features: torch.Tensor,
    ) -> torch.Tensor:
        normed_tokens = self.input_layernorm(token_embeddings)
        normed_conditioning = self.conditioning_feature_ln(conditioning_features)
        hidden_states = torch.cat([normed_tokens, normed_conditioning], dim=-1)
        
        hidden_states = self.self_attn(positions, hidden_states) 
        # use conditioning features as residual stream, not token embeddings, as per SAFEAILab ref impl
        hidden_states, residual = self.post_attention_layernorm(hidden_states, conditioning_features) 
        hidden_states = self.mlp(hidden_states) + residual
        return hidden_states 

class Eagle3DraftModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,  # Changed from Qwen3Config
        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        use_eagle: bool = False,
        eagle_layers: list[int] | None = None,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.draft = draft
        self.speculate = speculate
        self.spec_k = spec_k
        self.async_fan_out = async_fan_out
        self.draft_async = draft_async
        self.use_eagle = use_eagle
        self.eagle_layers = eagle_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            draft_async=self.draft_async,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        assert config.num_hidden_layers == 1, "ERROR in Eagle3DraftModel: config.num_hidden_layers must be 1"
        self.layer = Eagle3DecoderLayer(
            config,
            draft=self.draft,
            speculate=self.speculate,
            spec_k=self.spec_k,
            async_fan_out=self.async_fan_out,
            draft_async=self.draft_async,
            tp_group=tp_group,
            tp_size=tp_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states_projected: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = self.embed_tokens(input_ids)
        hidden_states = self.layer(positions, token_embeddings, target_hidden_states_projected)
        return hidden_states

class Eagle3DraftForCausalLM(nn.Module):
    packed_modules_mapping = {
        "midlayer.self_attn.q_proj": ("model.layer.self_attn.qkv_proj", "q"),
        "midlayer.self_attn.k_proj": ("model.layer.self_attn.qkv_proj", "k"),
        "midlayer.self_attn.v_proj": ("model.layer.self_attn.qkv_proj", "v"),
        "midlayer.mlp.gate_proj": ("model.layer.mlp.gate_up_proj", 0),
        "midlayer.mlp.up_proj": ("model.layer.mlp.gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,  
        draft: bool = False,
        speculate: bool = False,
        use_eagle: bool = False,
        eagle_layers: list[int] | None = None,
        d_model_target: int = 4096,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
        debug_mode: bool = False,
    ) -> None:
        super().__init__()

        assert draft, "ERROR in Eagle3DraftForLlama3: draft must be True"
        assert use_eagle, "ERROR in Eagle3DraftForLlama3: config.use_eagle must be True"
        assert eagle_layers is not None, "ERROR in Eagle3DraftForLlama3: eagle_layers must be set"

        # this will be the draft that does tree decode, just needs a modified fwd pass that takes in hidden states and uses fc and dicts to sample, etc 
        self.config = config
        self.draft = draft
        self.async_fan_out = async_fan_out
        self.draft_async = draft_async
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.use_eagle = use_eagle
        self.eagle_layers = eagle_layers if eagle_layers is not None else []
        self.d_model_target = d_model_target
        self.d2t = {}  # loaded by loader.py, converted to tensor after load_model
        self.t2d = {}  # loaded by loader.py, converted to tensor after load_model
        self.d2t_tensor = None  # will be set after load_model
        self.t2d_tensor = None  # will be set after load_model
        self.debug_mode = debug_mode
        self._debug_saved = False  # Track if we've already saved debug data
        assert not (tp_group is None and self.tp_size > 1), "ERROR in LlamaForCausalLM: tp_group is None and tp_size > 1"

        print(f'Starting Eagle3DraftForCausalLM init, draft={draft}, speculate={speculate}, spec_k={spec_k}')
        self.fc = nn.Linear(len(self.eagle_layers) * d_model_target, config.hidden_size, bias=False)
        self.final_norm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.model = Eagle3DraftModel(config, draft, speculate, spec_k, async_fan_out, draft_async, use_eagle=use_eagle, eagle_layers=eagle_layers, tp_group=tp_group, tp_size=self.tp_size)
        self.lm_head = ParallelLMHead(
            config.draft_vocab_size,  # LM head size (subset of tokens draft can propose)
            config.hidden_size,
            draft_async=draft_async,
            tp_group=tp_group,
            tp_size=self.tp_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        print(f'Finishing Eagle3DraftForCausalLM init, draft={draft}, speculate={speculate}, spec_k={spec_k}') 

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Only project if this is target hidden states (3 * d_model_target dimension)
        if hidden_states.shape[-1] == 3 * self.d_model_target:
            # This is the first prefill with target activations
            if self.debug_mode and not self._debug_saved and input_ids.shape[0] != 2048:
                self._save_debug_inputs(input_ids, positions, hidden_states)
                self._debug_saved = True
            
            hidden_states_projected = self.fc(hidden_states.to(self.fc.weight.dtype))  # [num_tokens, d_model_draft]
        else:
            hidden_states_projected = hidden_states # draft self-conditioning output, already d_model_draft from prenorm 
        
        # Forward through draft model with conditioning
        prenorm = self.model(input_ids, hidden_states_projected, positions)
        return prenorm
    
    def _save_debug_inputs(self, input_ids: torch.Tensor, positions: torch.Tensor, target_hidden_states: torch.Tensor):
        """Save draft prefill inputs for debugging."""
        import os
        # Get token embeddings
        with torch.no_grad():
            token_embeddings = self.model.embed_tokens(input_ids)
        
        debug_data = {
            'input_ids': input_ids.cpu(),
            'positions': positions.cpu(),
            'target_hidden_states': target_hidden_states.cpu(),  # [num_tokens, 3 * d_model_target]
            'token_embeddings': token_embeddings.cpu(),
            'd_model_target': self.d_model_target,
            'eagle_layers': self.eagle_layers,
        }
        
        os.makedirs('debug_outputs', exist_ok=True)
        save_path = 'debug_outputs/draft_prefill_inputs.pt'
        torch.save(debug_data, save_path)
        print(f"[DEBUG] Saved draft prefill inputs to {save_path}")
        print(f"[DEBUG] Shapes: input_ids={input_ids.shape}, positions={positions.shape}, target_hidden_states={target_hidden_states.shape}, token_embeddings={token_embeddings.shape}")


    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        last_only: bool = True, 
    ) -> torch.Tensor:
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states, last_only=last_only)  # [B, draft_vocab_size]

        if logits.dim() == 3:
            logits = logits.view(-1, logits.shape[-1])
        
        # Expand draft vocab logits to full target vocab using d2t mapping
        # Draft LM head has draft_vocab_size rows, map them to target vocab positions
        assert self.d2t_tensor is not None, "d2t_tensor must be loaded before inference"
        assert hasattr(self.config, 'vocab_size'), "config must have vocab_size (target vocab)"
        assert hasattr(self.config, 'draft_vocab_size'), "config must have draft_vocab_size"
        
        B = logits.shape[0]
        vocab_size = self.config.vocab_size  # Target vocab size from config
        
        # Map draft indices to target vocab positions: target_idx = draft_idx + d2t_tensor[draft_idx]
        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        target_indices = base + self.d2t_tensor  # [draft_vocab_size]
        # target_indices = self.d2t_tensor  # [draft_vocab_size]
        # Debug logging once per run
        if not hasattr(self, '_vocab_debug_printed'):
            print(
                f'[compute_logits DEBUG] draft_vocab_size={self.config.draft_vocab_size}, target_vocab_size={vocab_size}', flush=True)
            print(
                f'[compute_logits DEBUG] d2t_tensor[:10]={target_indices[:10].tolist()}', flush=True)
            print(
                f'[compute_logits DEBUG] d2t_tensor[-10:]={target_indices[-10:].tolist()}', flush=True)
            self._vocab_debug_printed = True
        
        # Scatter draft logits into full vocab space, -inf elsewhere
        logits_full = logits.new_full((B, vocab_size), float('-inf'))
        logits_full[:, target_indices] = logits
        
        return logits_full 

''' 
----- eagle3 draft weights for target=llama 3.1 8b ----- 
Weights in pytorch_model.bin:
  d2t: torch.Size([32000])
  t2d: torch.Size([128256])
  midlayer.self_attn.q_proj.weight: torch.Size([4096, 8192])      model.layer.self_attn.qkv_proj.weight
  midlayer.self_attn.k_proj.weight: torch.Size([1024, 8192])
  midlayer.self_attn.v_proj.weight: torch.Size([1024, 8192])
  midlayer.self_attn.o_proj.weight: torch.Size([4096, 4096])
  midlayer.mlp.gate_proj.weight: torch.Size([14336, 4096])
  midlayer.mlp.up_proj.weight: torch.Size([14336, 4096])
  midlayer.mlp.down_proj.weight: torch.Size([4096, 14336])
  midlayer.hidden_norm.weight: torch.Size([4096])
  midlayer.input_layernorm.weight: torch.Size([4096])
  midlayer.post_attention_layernorm.weight: torch.Size([4096])
  norm.weight: torch.Size([4096])
  fc.weight: torch.Size([4096, 12288])
  lm_head.weight: torch.Size([32000, 4096])
----- 

EAGLE3 control flow - one round of speculation 

- Target concats 3 layer preactivations at last accepted token (last token on prefill, bonus token on decode) into a [num_tokens, 3 * d_model_target] tensor    
    - This is done at layers [2, num_layers//2, num_layers-3] by default 
- Sends to draft over NCCL as part of the handshake protocol (all sequence positions in prefill, list of single tokens in decode)
If PREFILL
    - Draft receives [num_tokens, 3 * d_model_target]
    - Draft projects using FC to get [num_tokens, d_model_draft]
    - Draft fwd takes in [num_tokens, 2 * d_model_draft] as we cat[last_prefill_token_activation_projected; next_token_id_from_prefill_embedded]
    - The attn layer is is 2D -> D so that MLP still sees dimension D 
    - Draft KV cache updated (target fires and forgets in our codebase) with inputs that includes target activations 
If DECODE 
    - Draft projects to d_model_draft via fc for first token on every speculation round (ie. iter 0 of tree decode)
        - Uses its own lm_head preactivations which are d_model_draft as conditioning input in later iter > 0 of tree decode 
    - Do similar to prefill where we draft fwd but decode so num_tokens = B (num seqs being decoded)
    - Keep prenorms before lm_head as well as logits after lm_head, use the former as input to the next iter and the latter for sampling the token also for the next iter 
    - Draft always operates on target vocab token IDs. Draft LM head outputs draft_vocab_size logits, expanded to vocab_size in compute_logits via d2t mapping 


Impl plan: 
    - modify LlamaModel to take in hidden states, cat two things, output logits and prelogits for next round
        - make sure attn layer is 2D -> D (eg. by wiring in cfg), hook up to an attnLayer and a kv cache 
        - test decoding with correct shapes, double check arch and norms, ie. if we get activations we can decode just fine 
    - add draft weight loader support 
    - wire through config and init support in ModelRunner/LLMEngine (support only greedy decoding for now)
        - if cfg.use_eagle, then the target should collect activations 
    - support prefill, collecting and sending all activations via fire and forget 
    - add support for sending just single activations in handshake 


Details to check 
    - add vs cat of conditioning vector vs just wiring it into residual stream directly 
    - draft kv cache handling 
    - correct target activation layers and concatentation 
    - make sure eagle draft arch is exactly correct (see vLLM issue, pay attention to layernorm positioning, 
        see https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/modeling_llama_kv.py etc to check our arch/weights loaded are consistent)
    - tracking draft activations for self-conditioning in tree decode 
    - glue decode, may need to store previous iter draft activations for self-conditioning 
    - torch compile / cudagraph support for draft fwd / capturing 
'''
