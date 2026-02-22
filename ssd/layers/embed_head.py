import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from ssd.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__()

        self.draft_async = draft_async
        self.tp_group = tp_group
        self.tp_size = tp_size

        assert not (tp_group is None and self.tp_size > 1), "ERROR in VocabParallelEmbedding: tp_group is None and tp_size > 1"
        if self.tp_size > 1:
            # target shards [0, N-2] during draft_async get tp_group, self.tp_rank=N-1 then
            self.tp_rank = dist.get_rank(group=self.tp_group)
        else:
            # normal decoding or we are draft 
            self.tp_rank = 0

        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y, group=self.tp_group)
        # print(f'in vocab parallel embedding, shape of input: {x.shape}, shape of output: {y.shape}') # [nt] -> [nt, D]
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        draft_async: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        assert not bias, "ERROR in ParallelLMHead: bias is not supported"
        super().__init__(num_embeddings, embedding_dim, draft_async, tp_group, tp_size)
        self.draft_async = draft_async
        self.tp_group = tp_group
        self.tp_size = tp_size

    def forward(self, x: torch.Tensor, last_only: bool = True): # x is always [nt = B*S, D] -> [nt, V]
        context = get_context()
        if context.cu_seqlens_q is not None:  # mq decode (prefill, glue, verify, tree decode)
            if context.is_prefill:
                if last_only:
                    # [nt, D] -> [b, D] which later becomes [b, V]
                    last_indices = context.cu_seqlens_q[1:] - 1
                    x = x[last_indices].contiguous()
                else:
                    # Return logits for all tokens in prefill
                    flat_logits = F.linear(x, self.weight)
                    if self.tp_size > 1:
                        parts = [torch.empty_like(flat_logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
                        dist.gather(flat_logits, parts, 0, group=self.tp_group)
                        flat_logits = torch.cat(parts, dim=-1) if self.tp_rank == 0 else None
                    return flat_logits
            else: # multi-query decode path (glue, verify, tree)
                flat_logits = F.linear(x, self.weight)
                if self.tp_size > 1:
                    parts = [torch.empty_like(flat_logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
                    dist.gather(flat_logits, parts, 0, group=self.tp_group)
                    flat_logits = torch.cat(parts, dim=-1) if self.tp_rank == 0 else None
                if flat_logits is None:
                    return None
                # Check if constant query len (verify/tree) or variable (glue)
                batch_size = context.cu_seqlens_q.size(0) - 1
                total_tokens = x.size(0)
                if total_tokens % batch_size == 0:
                    constant_query_len = total_tokens // batch_size
                    return flat_logits.view(batch_size, constant_query_len, flat_logits.size(-1))
                return flat_logits  # variable-length: return flat [N, V]

        # decode, get single token 
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0, group=self.tp_group)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits

