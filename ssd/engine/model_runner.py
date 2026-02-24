
import pickle
import time
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from transformers import AutoTokenizer, AutoConfig
import os
import flashinfer
from ssd.config import Config
from ssd.engine.sequence import Sequence
from ssd.models.qwen3 import Qwen3ForCausalLM
from ssd.models.llama3 import LlamaForCausalLM
from ssd.models.eagle3_draft_llama3 import Eagle3DraftForCausalLM
from ssd.layers.sampler import Sampler
from ssd.utils.context import set_context, reset_context, get_context
from ssd.utils.loader import load_model
from ssd.engine.helpers.runner_helpers import (
    prepare_decode_tensors_from_seqs, 
    prepare_block_tables_from_seqs, 
    prepare_prefill_tensors_from_seqs
)
from ssd.engine.helpers.cudagraph_helpers import (
    run_verify_cudagraph,
    run_decode_cudagraph,
    run_fi_tree_decode_cudagraph,
    run_glue_decode_cudagraph,
    capture_cudagraph,
    capture_verify_cudagraph,
    capture_fi_tree_decode_cudagraph,
    capture_glue_decode_cudagraph,
    get_custom_mask,
)
    

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event], is_draft: bool = False, num_tp_gpus: int = -1, init_q = None):
        if config.verbose: print(f'ModelRunner init got args: rank={rank}, is_draft={is_draft}, num_tp_gpus={num_tp_gpus}', flush=True)
        self.config = config
        
        assert is_draft in [True, False], "ERROR in ModelRunner: is_draft must be True or False"
        self.is_draft = is_draft
        if self.is_draft: 
            if config.draft_hf_config.torch_dtype != config.hf_config.torch_dtype:
                if self.verbose:
                    print(f"Warning: Draft dtype {config.draft_hf_config.torch_dtype} differs from target {config.hf_config.torch_dtype}. Casting draft to {config.hf_config.torch_dtype}.")
                config.draft_hf_config.torch_dtype = config.hf_config.torch_dtype
            assert (config.draft_hf_config.vocab_size == config.hf_config.vocab_size) or config.use_eagle, "ERROR in ModelRunner: draft_hf_config.vocab_size != hf_config.vocab_size"
        
        self.hf_config = config.hf_config if not is_draft else config.draft_hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path if config.tokenizer_path else config.model, use_fast=True)
        self.max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        assert self.hf_config is not None, "ERROR in ModelRunner: hf_config is None" # this implies boundedness to the end 
        
        # TODO: Get rid of this.
        if self.is_draft:
            should_use_dist = self.config.draft_async
        else:
            should_use_dist = self.config.num_gpus > 1

        # determines whether we create a process group and our process is aware of others, etc
        self.world_size = config.num_gpus if should_use_dist else 1
        self.rank = rank
        self.use_eagle = config.use_eagle

        if config.draft_async:
            self.draft_rank = config.num_gpus - 1
            if self.is_draft:
                # [N, 3] int64, {(seq_id, len(new_suffix), rec_token): (speculated_tokens, logits_q)}
                self.prev_fork_keys: torch.Tensor | None = None
                self.prev_fork_block_tables: torch.Tensor | None = None  # [N, M] int32 with -1 padding

        self.num_tp_gpus = num_tp_gpus # will be in [1 if no dist, num_gpus if not async, num_gpus-1 if async]
        
        if self.world_size == 1: # sync speculation or genuine single gpu 
            assert (config.speculate and not config.draft_async) or self.num_tp_gpus == 1, "ERROR in ModelRunner: draft and async must be False or num_tp_gpus=1"

        self.verbose = config.verbose
        self.draft_async = config.draft_async
        self.event = event
        self._exiting = False 
        
        torch.cuda.set_device(self.rank)
        self.device = torch.device(f'cuda:{self.rank}') 
        
        # cudagraph logic for FlashInfer kernels, need diff wrapper for each batch size we make a graph for 
        if is_draft and config.draft_async:
            self._init_flashinfer_wrappers()
        
        if self.verbose: print(f'INSIDE MODEL RUNNER INIT, DRAFT={is_draft}', flush=True)
        self.tp_pg = None 

        if should_use_dist: 
            default_port = 1223 
            dist.init_process_group(
                "nccl", f"tcp://localhost:{default_port}",
                world_size=self.world_size,
                rank=self.rank,
                device_id=self.device,
            )

            self.tp_pg = dist.new_group(ranks=list(range(self.num_tp_gpus))) # everyone should see the new_group init even if not in group 

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        if self.is_draft:
            assert num_tp_gpus == 1, "ERROR in ModelRunner: draft should have tp_size=1"
            self.tp_pg = None # every rank is given an object from self.tp_pg, even tho draft doesnt participate it gets GROUP_NON_MEMBER object != None back, so we can't assert None here, we 
        
        print(f'[model_runner] about to setup and warmup model and cudagraphs, is use_eagle={self.use_eagle}', flush=True)
        model_type = self.setup_and_warmup_model_and_cudagraphs(config, self.hf_config, init_q, is_draft)

        if self.verbose: print(f'-----CAPTURED {model_type}CUDAGRAPH----', flush=True)
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.config.draft_async:
            if self.config.fan_out_list is None:
                self.config.fan_out_list = [
                    self.config.async_fan_out] * (self.config.speculate_k + 1)

            self.config.fan_out_t = torch.tensor(self.config.fan_out_list, device=self.device)
            self.config.fan_out_t_miss = torch.tensor(self.config.fan_out_list_miss, device=self.device)
            assert len(self.config.fan_out_list) == self.config.speculate_k + \
                1, "ERROR in Config: fan_out_list must be length speculate_k + 1"
            assert any(f > 0 for f in self.config.fan_out_list), "ERROR in Config: fan_out_list must be > 0"
            self.config.MQ_LEN = sum(self.config.fan_out_list)
            print(f'F={self.config.async_fan_out}, fan_out_list={self.config.fan_out_list}, fan_out_list_miss={self.config.fan_out_list_miss}, MQ_LEN={self.config.MQ_LEN}', flush=True)

        if should_use_dist: # (draft model when async=False or just single gpu logic) doesn't even enter this loop 
            if self.is_draft and self.draft_async: 
                pass # handled on draft runner after this init, includes doing draft_loop
            elif self.rank == 0: # target in a distributed setup 
                # Try to clean up any existing shared memory first
                try:
                    existing_shm = SharedMemory(name="ssd")
                    existing_shm.close() # here we bind it 
                    existing_shm.unlink()
                except FileNotFoundError:
                    # can proceed, nothing to clean up 
                    pass
                
                self.shm = SharedMemory(name="ssd", create=True, size=2**28)
                dist.barrier(group=self.tp_pg, device_ids=[self.rank]) # leader on tp_group 
            else: 
                dist.barrier(group=self.tp_pg, device_ids=[self.rank]) # follower on tp_group, don't want them hooking onto shm before its been created 
                self.shm = SharedMemory(name="ssd")
                self.loop()
                
        if self.verbose: print(f'-----{model_type}MODEL RUNNER INITIALIZED----', flush=True)

    def _init_flashinfer_wrappers(self):
        """Initialize FlashInfer wrappers for draft async mode."""
        self.workspace_buffer = torch.zeros(
            512 * 1024 * 1024, dtype=torch.uint8, device=f"cuda:{self.rank}") 
        
        if self.config.enforce_eager: 
            self.only_prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
        else: 
            max_bs = min(self.config.max_num_seqs, 512)
            max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
            
            # FlashInfer kernel tensors
            # pages_for_max_len = (self.config.max_model_len + self.block_size - 1) // self.block_size
            last_page_len_max_len = self.config.max_model_len % self.block_size
            last_page_len_max_len = self.block_size if last_page_len_max_len == 0 else last_page_len_max_len
            MQ_LEN = self.config.async_fan_out * (self.config.speculate_k + 1)
            
            cu_seqlens_q = torch.empty(max_bs + 1, dtype=torch.int32, device=self.device)
            kv_indptr = torch.empty(max_bs + 1, dtype=torch.int32, device=self.device)
            kv_indices = torch.empty(max_bs * max_num_blocks, dtype=torch.int32, device=self.device)
            kv_last_page_len = torch.empty(max_bs, dtype=torch.int32, device=self.device)
            custom_mask_buf = torch.empty(max_bs * MQ_LEN * self.config.max_model_len, dtype=torch.uint8, device=self.device)
            mask_indptr_buf = torch.empty(max_bs + 1, dtype=torch.int32, device=self.device)
            
            # Create graph_bs_list to match what will be used in cudagraph_helpers.py
            graph_bs_list = [1]
            for bs in [2, 4, 8] + list(range(16, max_bs + 1, 16)):
                if bs <= max_bs:
                    graph_bs_list.append(bs)
            if max_bs not in graph_bs_list:
                graph_bs_list.append(max_bs)
            graph_bs_list.sort()
            
            # Create a dict of wrappers, one for each bs we will touch in cudagraph_helpers.py
            self.prefill_wrappers = {}
            print(f'[model_runner about to wrapper.init()] graph_bs_list={graph_bs_list}', flush=True)
            for bs in graph_bs_list:
                self.prefill_wrappers[bs] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer, "NHD", 
                    use_cuda_graph=True, 
                    qo_indptr_buf=cu_seqlens_q[:bs + 1],
                    paged_kv_indptr_buf=kv_indptr[:bs + 1],
                    paged_kv_indices_buf=kv_indices[:bs * max_num_blocks],
                    paged_kv_last_page_len_buf=kv_last_page_len[:bs],
                    custom_mask_buf=custom_mask_buf[:bs * MQ_LEN * self.config.max_model_len],
                    mask_indptr_buf=mask_indptr_buf[:bs + 1],
                )
            print(f'wrapper backend is {self.prefill_wrappers[bs]._backend}', flush=True)


    def setup_and_warmup_model_and_cudagraphs(self, config: Config, hf_config: AutoConfig, init_q=None, is_draft=False):
        # cudagraphs 
        self.graph_vars = {}
        self.graph_pools = {}
        self.graphs = {}
        self.graph_bs_list = {}

        assert hasattr(hf_config, 'model_type'), "ERROR in ModelRunner: hf_config.model_type is not set"
        if config.use_eagle and is_draft:
            print(f'[EAGLE3] Loading Eagle3DraftForCausalLM as model_class', flush=True)
            model_class = Eagle3DraftForCausalLM
        elif hf_config.model_type == 'llama':
            model_class = LlamaForCausalLM
        elif hf_config.model_type == 'qwen3':
            model_class = Qwen3ForCausalLM
        else:
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")

        # only give Qwen3 a tp_group if we're a TARGET runner
        kwargs = dict(
            config=self.hf_config,
            draft=self.is_draft,
            speculate=self.config.speculate,
            spec_k=self.config.speculate_k,
            async_fan_out=self.config.async_fan_out,
            draft_async=self.config.draft_async,
            tp_group=self.tp_pg,
            tp_size=self.num_tp_gpus,
        )
        
        if config.use_eagle:
            kwargs['use_eagle'] = True
            kwargs['eagle_layers'] = self.config.eagle_layers
            
        if model_class == Eagle3DraftForCausalLM:
            kwargs['d_model_target'] = config.d_model_target
            kwargs['debug_mode'] = config.debug_mode
            
        self.model = model_class(**kwargs)

        model_type = "DRAFT " if self.is_draft else "TARGET "
        if self.verbose:
            print(f'-----LOADING {model_type}MODEL----', flush=True)
        
        # Pass tokenizer_path as target_path if it's available (mostly for Eagle draft)
        target_path = getattr(config, 'tokenizer_path', None)
        target_hidden_size = getattr(config, 'd_model_target', None)
        load_model(self.model, config.model, target_path=target_path, target_hidden_size=target_hidden_size)
        
        if config.draft_async:  # move this here so we don't get a timeout waiting for draft rank while load_model happens?
            self.async_pg = dist.new_group(ranks=[0, self.draft_rank])
        if self.verbose:
            print(f'-----{model_type}MODEL LOADED----', flush=True)
        if config.sampler_x is not None:
            assert config.draft_async, "ERROR in ModelRunner: sampler_x requires draft_async"
            assert config.draft_async_temp == config.target_async_temp, "ERROR in ModelRunner: draft_async_temp != target_async_temp"
            assert sum(config.fan_out_list) == sum(config.fan_out_list_miss) == config.async_fan_out * (config.speculate_k + 1), "ERROR in ModelRunner: fancy sampling only supported for constant fan out for now."

        self.sampler = Sampler(sampler_x=config.sampler_x, async_fan_out=config.async_fan_out)
        if self.verbose:
            print(f'-----WARMING UP {model_type}MODEL----', flush=True)
        self.warmup_model()
        if self.verbose:
            print(f'-----ALLOCATING {model_type}KV CACHE----', flush=True)
        self.allocate_kv_cache()
        if init_q is not None:
            # super().__init__() runs warmup and calculates num_kvcache_blocks, pass that up
            init_q.put(self.config.num_kvcache_blocks)
            init_q.close()

        if not self.enforce_eager:
            # if not self.is_draft or (self.is_draft and self.config.draft_async and self.config.speculate): 
            decode_graph_vars, decode_graph_pool, decode_graphs, decode_graph_bs_list = capture_cudagraph(self)  # decode cudagraph, draft needs in spec and target in normal
            self.graph_vars["decode"] = decode_graph_vars
            self.graph_pools["decode"] = decode_graph_pool
            self.graphs["decode"] = decode_graphs
            self.graph_bs_list["decode"] = decode_graph_bs_list
            if self.config.speculate and not (self.is_draft and self.config.use_eagle):  # verify CG: target always, non-EAGLE draft for fan-out; EAGLE draft uses glue_decode CG instead
                verify_graph_vars, verify_graph_pool, verify_graphs, verify_graph_bs_list = capture_verify_cudagraph(self)
                self.graph_vars["verify"] = verify_graph_vars
                self.graph_pools["verify"] = verify_graph_pool
                self.graphs["verify"] = verify_graphs
                self.graph_bs_list["verify"] = verify_graph_bs_list
            if self.config.speculate and self.is_draft and self.config.draft_async:
                fi_tree_decode_graph_vars, fi_tree_decode_graph_pool, fi_tree_decode_graphs, fi_tree_decode_graph_bs_list = capture_fi_tree_decode_cudagraph(self)  # fi tree decode cudagraph, draft only
                self.graph_vars["fi_tree_decode"] = fi_tree_decode_graph_vars
                self.graph_pools["fi_tree_decode"] = fi_tree_decode_graph_pool
                self.graphs["fi_tree_decode"] = fi_tree_decode_graphs
                self.graph_bs_list["fi_tree_decode"] = fi_tree_decode_graph_bs_list
            if self.config.speculate and self.is_draft and self.config.draft_async and self.config.use_eagle:
                glue_gv, glue_pool, glue_graphs, glue_bs_list = capture_glue_decode_cudagraph(self)
                self.graph_vars["glue_decode"] = glue_gv
                self.graph_pools["glue_decode"] = glue_pool
                self.graphs["glue_decode"] = glue_graphs
                self.graph_bs_list["glue_decode"] = glue_bs_list

        return model_type

    def exit(self, hard: bool = True):
        # Idempotent
        if getattr(self, "_exiting", False):
            return
        self._exiting = True
        if self.verbose:
            print(f"[ModelRunner] Exit start (rank={self.rank}, draft={self.is_draft}, hard={hard})", flush=True)
        # 1) Notify draft first (target rank 0 only)
        try:
            self.send_draft_exit_signal()
        except Exception:
            pass
        # 2) Best-effort local cleanup (no collectives; avoid group destroys in hard mode)
        try:
            if not self.enforce_eager and hasattr(self, "graphs"):
                del self.graphs
                if hasattr(self, "graph_pool"):
                    del self.graph_pool
            if hasattr(self, "verify_graphs"):
                del self.verify_graphs
            if hasattr(self, "verify_graph_pool"):
                del self.verify_graph_pool
            if hasattr(self, "glue_graphs"):
                del self.glue_graphs
            if hasattr(self, "glue_graph_pool"):
                del self.glue_graph_pool
        except Exception:
            pass
        # Close SHM on all ranks that have it
        try:
            if hasattr(self, "shm") and self.shm is not None:
                self.shm.close()
                if self.rank == 0:
                    try:
                        self.shm.unlink()
                    except Exception:
                        pass
        except Exception:
            pass
        # 3) Non-hard path: try to tear down process groups (may block, so do only if not hard)
        if not hard:
            try:
                if hasattr(self, "async_pg"):
                    dist.destroy_process_group(self.async_pg)
            except Exception:
                pass
            try:
                if self.world_size > 1 and hasattr(self, "tp_pg") and self.tp_pg is not None:
                    dist.destroy_process_group(self.tp_pg)
            except Exception:
                pass
            try:
                # Default group
                if self.world_size > 1 or (self.draft_async and self.is_draft):
                    dist.destroy_process_group()
            except Exception:
                pass
            return
        # 4) Hard path: do not destroy groups (avoid NCCL waits). Draft process can exit immediately.
        if self.is_draft:
            if self.verbose:
                print(f"[ModelRunner] Draft hard-exit", flush=True)
            os._exit(0)
        # Target ranks: let the process return if we're a worker (subprocess),
        # main rank will be force-exited by the engine after it joins children.
        if self.verbose:
            print(f"[ModelRunner] Exit complete (rank={self.rank})", flush=True)
        return

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def recv_cmd(self):
        t = torch.empty(1, dtype=torch.int64, device=self.device)
        dist.recv(t, src=0, group=self.async_pg)
        return int(t.item())

    def recv_tensor(self, shape, dtype=torch.int64):
        t = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(t, src=0, group=self.async_pg)
        return t
    
    def send_draft_exit_signal(self):
        """
        Best-effort: send cmd=2 to draft (async_pg) from TARGET rank 0.
        Safe to call multiple times.
        """
        if not (self.draft_async and not self.is_draft and self.rank == 0):
            return
        try:
            cmd = torch.tensor([2], dtype=torch.int64, device=self.device)
            dist.send(cmd, dst=self.draft_rank, group=self.async_pg)
        except Exception:
            pass

    def _wait_for_cmd(self, handle_entry):
        """Waits for a command, using the provided handle if available."""
        if handle_entry:
            work_handle, cmd_tensor = handle_entry
            # block until the irecv completes and the buffer is filled
            work_handle.wait()
            return int(cmd_tensor.item()), None
        else:
            # no pending irecv, fall back to the normal recv path
            return self.recv_cmd(), None

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        assert n + 4 <= self.shm.size, f"SHM overflow: {n+4} > {self.shm.size}. Increase SHM buffer size."
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        if method is None:
            raise AttributeError(f"Method '{method_name}' not found")
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        
        hidden_states = None
        if self.config.use_eagle and self.is_draft:
            num_tokens = num_seqs * max_model_len
            d_model_target = self.config.d_model_target or 4096
            hidden_states = torch.zeros(num_tokens, 3 * d_model_target, dtype=self.hf_config.torch_dtype, device=self.device)
        
        self.run(seqs, True, hidden_states=hidden_states)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        print(f'inside allocate_kv_cache -- ', flush=True)
        config = self.config
        hf_config = self.hf_config
        
        # Simplify: just look at free memory on the GPU, and allocate up to gpu_memory_utilization * free.
        free, _ = torch.cuda.mem_get_info()
        num_kv_heads = hf_config.num_key_value_heads // self.num_tp_gpus
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * hf_config.head_dim
            * hf_config.torch_dtype.itemsize
        )
        usable_bytes = free * config.gpu_memory_utilization

        if self.is_draft and self.draft_async:
            B = config.max_num_seqs          # max concurrent sequences
            K = config.speculate_k           # speculative steps
            F = config.async_fan_out         # branches per step
            V = hf_config.vocab_size
            dtype_size = hf_config.torch_dtype.itemsize
            # Total forks = B * (K+1) * F, each fork stores (K+1)*V logits of size dtype_size
            reserved_bytes = B * (K + 1) * F * (K + 1) * V * dtype_size
            usable_bytes = max(usable_bytes - reserved_bytes, 0)
            assert usable_bytes > 0, "ERROR: Not enough memory for draft KV cache after accounting for tree_cache for logits storage"

        config.num_kvcache_blocks = int(usable_bytes) // block_bytes
        if self.verbose:
            print(f'KV CACHE ALLOCATION for {"TARGET" if not self.is_draft else "DRAFT"} model', flush=True)
            print(f' free={free/1e9:.2f}GB, util={config.gpu_memory_utilization:.2f}', flush=True)
            print(f' block_bytes={block_bytes/1e9:.2f}GB,'
            + f' num_kvcache_blocks={config.num_kvcache_blocks}', flush=True)   
        
        assert config.num_kvcache_blocks > 0, "KV cache too big for free memory!"

        self.kv_cache = torch.zeros( 
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            hf_config.head_dim, 
        )
        
        print(f"allocate_kv_cache(): kv_cache shape = {self.kv_cache.shape}", flush=True)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                if self.is_draft and self.draft_async and not self.enforce_eager:
                    module.prefill_wrappers = self.prefill_wrappers
                elif self.is_draft and self.draft_async and self.enforce_eager:
                    module.only_prefill_wrapper = self.only_prefill_wrapper # this will make it not None so it can be used on fwd
                layer_id += 1

    
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids, positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping = \
            prepare_prefill_tensors_from_seqs(seqs, self.block_size, self.is_draft) # if one big input ids, how is attn mask handled? via cu_seqlens_q/k? 

        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = prepare_block_tables_from_seqs(
                seqs, self.is_draft)
        
        set_context(is_prefill=True, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k, slot_mapping=slot_mapping, context_lens=None, block_tables=block_tables)
        return input_ids, positions
    
    def prepare_decode(self, seqs: list[Sequence], verify: bool = False): 
        input_ids, positions, slot_mapping, context_lens = \
            prepare_decode_tensors_from_seqs(seqs, self.block_size, self.is_draft, verify, self.config.speculate_k if verify else -1)

        
        block_tables = prepare_block_tables_from_seqs(seqs, self.is_draft) # if verify, set cu_seqlens_q as well

        if verify: ### what path does glue decode take? trace it. 
            # this had [not draft and draft_async] condn before
            cu_seqlens_q = torch.zeros(len(seqs) + 1, dtype=torch.int32, device=self.device)
            seqlen_q = torch.full((len(seqs),), self.config.speculate_k + 1, dtype=torch.int32, device=self.device)
            cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)
            set_context(is_prefill=False, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=None, 
                       max_seqlen_q=self.config.speculate_k + 1, max_seqlen_k=0,
                       slot_mapping=slot_mapping, context_lens=context_lens, 
                       block_tables=block_tables) 
        else: # sq_decode path, draft (sync spec) or target (normal)
            set_context(is_prefill=False, cu_seqlens_q=None, cu_seqlens_k=None, 
                       max_seqlen_q=0, max_seqlen_k=0, slot_mapping=slot_mapping, 
                       context_lens=context_lens, block_tables=block_tables)
        
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            if self.is_draft and seq.draft_temperature is not None:
                temperatures.append(seq.draft_temperature)
            else:
                temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def eager_tree_decode_plan(self, input_ids, positions, step, cache_hits):
        """Plan FlashInfer for tree decode in eager mode"""
        assert self.is_draft and self.config.draft_async, "ERROR in eager_tree_decode_plan: not a draft async model"
        context = get_context()
        
        K, F = self.config.speculate_k, self.config.async_fan_out
        # MQ_LEN = F * (K+1)
        MQ_LEN = self.config.MQ_LEN
        flat_batch_size = input_ids.size(0) 
        B = flat_batch_size // MQ_LEN # [N] tokens = B * sum(fan_out_list)
        
        # Convert block_tables to FlashInfer format
        block_tables = context.block_tables # [B, M]
        context_lens = context.context_lens # [B]
        
        counts = (context_lens + self.block_size - 1) // self.block_size # [B]
        kv_indptr = torch.cat([torch.tensor([0], device=block_tables.device),
                               counts.cumsum(dim=0)]).to(torch.int32)  
        mask = torch.arange(block_tables.size(1), device=block_tables.device)[None, :] < counts[:, None]
        kv_indices = block_tables[mask]                    # flattened page ids
        
        # Last-page actual token count per request
        kv_last_page_len = (context_lens % self.block_size)
        kv_last_page_len[kv_last_page_len == 0] = self.block_size
        kv_last_page_len = kv_last_page_len.to(torch.int32)
        cu_seqlens_q = torch.arange(B + 1, device=self.device, dtype=torch.int32) * MQ_LEN # assumes same MQ_LEN across batch dimension 
        custom_mask = get_custom_mask(self.config, context_lens, step, K, F, B, device=self.device, cache_hits=cache_hits)
        
        self.only_prefill_wrapper.plan(
            cu_seqlens_q,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            self.hf_config.num_attention_heads,
            self.hf_config.num_key_value_heads,
            self.hf_config.head_dim,
            self.block_size,
            custom_mask=custom_mask,
            q_data_type=self.hf_config.torch_dtype,
            kv_data_type=self.hf_config.torch_dtype,
        )

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, last_only: bool = True, tree_decode_step: int = -1, cache_hits: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None):
        is_tree_decode = self.is_draft and self.config.draft_async and tree_decode_step >= 0
        is_mq_kp1 = self.config.speculate and not last_only
        spec_and_dec = not is_prefill and self.config.speculate

        assert not (is_prefill and not last_only), "ERROR in run_model: is_prefill and not last_only"
        
        if is_prefill or self.enforce_eager:
            if is_tree_decode:
                self.eager_tree_decode_plan(input_ids, positions, tree_decode_step, cache_hits)
            
            if self.config.use_eagle: 
                if self.is_draft:
                    assert hidden_states is not None, "hidden_states required for EAGLE draft"
                    assert isinstance(self.model, Eagle3DraftForCausalLM)
                    prenorm = self.model(input_ids, positions, hidden_states)
                    logits = self.model.compute_logits(prenorm, last_only)
                    return logits, prenorm  # return prenorm as conditioning vector for next iteration
                else: # target gets outputs and hidden states
                    outputs, eagle_acts = self.model(input_ids, positions) # target fwd when eagle enabled hooks into activations for eagle conditioning
                    logits = self.model.compute_logits(outputs, last_only)
                    return logits, eagle_acts  # return eagle_acts as conditioning vector for draft
            else: 
                outputs = self.model(input_ids, positions)
                logits = self.model.compute_logits(outputs, last_only)
                return logits 

        elif is_tree_decode:
            return run_fi_tree_decode_cudagraph(self, input_ids, positions, last_only, self.graph_vars["fi_tree_decode"], tree_decode_step, cache_hits, hidden_states=hidden_states)
        elif is_mq_kp1 and hidden_states is not None and "glue_decode" in self.graph_vars:
            # EAGLE draft glue decode with 2K+1 per seq
            return run_glue_decode_cudagraph(self, input_ids, positions, last_only, self.graph_vars["glue_decode"], hidden_states)
        elif is_mq_kp1: # verify or non-EAGLE glue decode, "verify" ~ mq decode of len K+1
            return run_verify_cudagraph(self, input_ids, positions, last_only, self.graph_vars["verify"])
        else: # draft decoding in sync spec or JIT single-token decode
            return run_decode_cudagraph(self, input_ids, positions, last_only, self.graph_vars["decode"], hidden_states=hidden_states)


    # should add spec_k that just loops this k times
    def run(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
        last_only: bool = True,
        draft_return_logits: bool = False,
        hidden_states: torch.Tensor | None = None
    ) -> list[int] | tuple[list[int], torch.Tensor]:
        _pt = os.environ.get("SSD_PROFILE_TARGET", "0") == "1" and not is_prefill and not last_only
        if _pt:
            torch.cuda.synchronize()
            _r0 = time.perf_counter()

        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs, verify=not last_only)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        if _pt:
            torch.cuda.synchronize()
            _r1 = time.perf_counter()

        # Handle EAGLE returning (logits, conditioning_vector for next iter)
        conditioning = None
        if self.config.use_eagle:
            logits, conditioning = self.run_model(
                input_ids, positions, is_prefill, last_only, hidden_states=hidden_states)
        else:
            logits = self.run_model(input_ids, positions, is_prefill, last_only, hidden_states=hidden_states)

        if _pt:
            torch.cuda.synchronize()
            _r2 = time.perf_counter()
            print(f"[PROFILE target_run] prepare_decode={(_r1-_r0)*1000:.2f}ms run_model={(_r2-_r1)*1000:.2f}ms eagle={self.config.use_eagle} n_ids={input_ids.shape[0]}", flush=True)

        if last_only:
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
            reset_context()
            if conditioning is not None:
                return token_ids, conditioning
            return (token_ids, logits) if draft_return_logits else token_ids
        else:
            reset_context()
            if conditioning is not None:
                return logits, conditioning
            return logits
    

