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


class LlamaAttention(nn.Module):  # Renamed from Qwen3Attention

    def __init__( 
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 500000,
        rope_scaling: dict | None = None,
        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
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
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,  # Llama doesn't use QKV bias
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
        
        # Llama 3 doesn't use rope scaling but 3.1 does (and Qwen3 does) -- this only makes a difference on long context prompts, which we don't test 
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
            F=async_fan_out,
            K=spec_k,
        )
        # no qk norm for llama3 compared to qwen3 

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output

        
class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            tp_group=tp_group,
            tp_size=tp_size,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class LlamaDecoderLayer(nn.Module):

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
    ) -> None:
        super().__init__() 
        self.draft = draft
        self.speculate = speculate
        self.spec_k = spec_k
        self.async_fan_out = async_fan_out
        self.draft_async = draft_async
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 500000),
            rope_scaling=getattr(config, "rope_scaling", None),
            draft=self.draft,
            speculate=self.speculate,
            spec_k=self.spec_k,
            async_fan_out=self.async_fan_out,
            draft_async=self.draft_async,
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
        self.post_attention_layernorm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # assert hidden_states.shape
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
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
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                config,
                draft=self.draft,
                speculate=self.speculate,
                spec_k=self.spec_k,
                async_fan_out=self.async_fan_out,
                draft_async=self.draft_async,
                tp_group=tp_group,
                tp_size=tp_size,
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSDNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)  # torch.Size([4096, 2560]) always through residual stream 
        residual = None
        
        # Collect activations if use_eagle
        collected_acts = [] if self.use_eagle else None
        
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            
            if collected_acts is not None: collected_acts.append(hidden_states)
        
        hidden_states, _ = self.norm(hidden_states, residual) 
        
        if collected_acts:
            eagle_acts = torch.cat(collected_acts, dim=-1)
            return hidden_states, eagle_acts
        else:
            return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,  # Changed from Qwen3Config
        draft: bool = False,
        speculate: bool = False,
        use_eagle: bool = False,
        eagle_layers: list[int] | None = None,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()

        # if this is the standalone draft process, we want tp_size==1
        self.draft = draft
        self.async_fan_out = async_fan_out
        self.draft_async = draft_async
        self.use_eagle = use_eagle
        self.eagle_layers = eagle_layers
        self.tp_group = tp_group
        self.tp_size = tp_size
        
        assert not (use_eagle and draft), "ERROR in LlamaForCausalLM: use_eagle should be on EagleDraftForCausalLM and not LlamaForCausalLM"
        assert not (tp_group is None and self.tp_size > 1), "ERROR in LlamaForCausalLM: tp_group is None and tp_size > 1"

        print(f'Starting LlamaForCausalLM init, draft={draft}, speculate={speculate}, spec_k={spec_k}')
        self.model = LlamaModel(config, draft, speculate, spec_k, async_fan_out, draft_async, use_eagle=use_eagle, tp_group=tp_group, tp_size=self.tp_size)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            draft_async=draft_async,
            tp_group=tp_group,
            tp_size=self.tp_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        print(f'Finishing LlamaForCausalLM init, draft={draft}, speculate={speculate}, spec_k={spec_k}') 

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.model(input_ids, positions)
        return out


    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        last_only: bool = True, 
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states, last_only=last_only)
        return logits
