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
from ssd.models.llama3 import LlamaModel, LlamaDecoderLayer


# should benchmark how fast a fwd is -- hopefully <1ms since it's only one layer! 

class Eagle3DraftModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,  
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
        self.speculate = speculate
        self.spec_k = spec_k
        self.async_fan_out = async_fan_out
        self.draft_async = draft_async
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
    ) -> torch.Tensor:
        # torch.Size([4096, 2560]) always through residual stream
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

class Eagle3DraftForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,  
        draft: bool = False,
        speculate: bool = False,
        spec_k: int = 1,
        async_fan_out: int = 1,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()

        assert draft, "ERROR in Eagle3DraftForLlama3: draft must be True"
        assert config.use_eagle, "ERROR in Eagle3DraftForLlama3: config.use_eagle must be True"

        # this will be the draft that does tree decode, just needs a modified fwd pass that takes in hidden states and uses fc and dicts to sample, etc 
        self.draft = draft
        self.async_fan_out = async_fan_out
        self.draft_async = draft_async
        self.tp_group = tp_group
        self.tp_size = tp_size
        
        assert not (tp_group is None and self.tp_size > 1), "ERROR in LlamaForCausalLM: tp_group is None and tp_size > 1"

        print(f'Starting LlamaForCausalLM init, draft={draft}, speculate={speculate}, spec_k={spec_k}')
        self.model = LlamaModel(config, draft, speculate, spec_k, async_fan_out, draft_async, tp_group=tp_group, tp_size=self.tp_size)
        self.lm_head = ParallelLMHead(
            config.draft_vocab_size, # TODO: this is different for eagle draft, ie. 32_000 vs 128_256, use draft_vocab
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
    ) -> torch.Tensor:
        # hidden_states is 3 * d_model_target concatenated, we want to project it via self.fc then self condition with inputs 
        hidden_states = self.model(input_ids, positions) # should go inside model and ignore fwd activations 
        return hidden_states


    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        last_only: bool = True, 
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states, last_only=last_only)
        return logits

    # def sample(draft_vocab_logits: torch.Tensor) -> list[int]: # needs to use d2t/t2d dicts 

''' 
----- eagle3 draft weights for target=llama 3.1 8b ----- 
Weights in pytorch_model.bin:
  d2t: torch.Size([32000])
  t2d: torch.Size([128256])
  midlayer.self_attn.q_proj.weight: torch.Size([4096, 8192])
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
    - Each model should only see/ingest tokens from its own vocab -- prefill tokens send to draft need to be map(t2d) first and vice versa when draft sends back its speculations to target 


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
''' 