import time
import torch
from collections import deque
from transformers import AutoTokenizer

from ssd.config import Config
from ssd.engine.sequence import Sequence, SequenceStatus
from ssd.engine.block_manager import BlockManager

from ssd.utils.async_helpers.async_spec_helpers import compute_megaspec_lookahead

class Scheduler:

    def __init__(self, config: Config, draft_cfg: Config | None = None):
        self.max_num_seqs = config.max_num_seqs
        self.fan_out_list = config.fan_out_list
        self.fan_out_list_miss = config.fan_out_list_miss
        if config.draft_async:
            self.MQ_LEN = sum(self.fan_out_list)
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos = config.eos
        self.speculate = config.speculate
        self.F = config.async_fan_out
        self.K = config.speculate_k
        self.block_size = config.kvcache_block_size
        self.verbose = config.verbose
        self.draft_async = config.draft_async
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size, is_draft=False, verbose=self.verbose, max_model_len=self.max_model_len)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model)

        # num_kvcache_blocks is determined by gpu_mem_allocation in allocate()
        if self.speculate:
            self.draft_block_manager = BlockManager(
                draft_cfg.num_kvcache_blocks, draft_cfg.kvcache_block_size, is_draft=True, speculate_k=self.K, verbose=self.verbose, max_model_len=self.max_model_len)

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq) # is the issue when f(k+1)>block_sz?

    def bms_can_append(self, seq: Sequence, target_lookahead_len: int, draft_lookahead_len: int | None = None) -> bool:
        target_can_append = self.block_manager.can_append(seq, target_lookahead_len)
        if self.speculate:
            draft_can_append = self.draft_block_manager.can_append(
                seq, draft_lookahead_len)
        else:
            assert draft_lookahead_len is None, "ERROR in bms_can_append: draft_lookahead_len should be None if not speculate"
            draft_can_append = True

        return target_can_append and draft_can_append

    def bms_can_allocate(self, seq: Sequence) -> bool:
        return self.block_manager.can_allocate(seq) and (not self.speculate or self.draft_block_manager.can_allocate(seq))

    # what if we added an option to prefill jit
    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_batched_tokens = 0 # within this round only 
        
        while self.waiting: 

            seq = self.waiting[0]

            # num tokens that are not yet in the kv cache, eg. can be <seq.num_tokens in case of prefix cache usage
            remain = len(seq) - seq.num_cached_tokens 

            if num_batched_tokens + remain > self.max_num_batched_tokens or not self.bms_can_allocate(seq):
                break 
            
            self.block_manager.allocate(seq)
            if self.speculate:
                self.draft_block_manager.allocate(seq)

            num_batched_tokens += remain

            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            if __debug__: print(f'[scheduler] returning {len(scheduled_seqs)} sequences for prefill', flush=True)
            return scheduled_seqs, True

        # decode, these sequences are already running
        num_seqs_decoded = 0 
        sync_spec = self.speculate and not self.draft_async
        async_spec = self.speculate and self.draft_async
        
        if async_spec:
            target_lookahead_len = self.K + 1
            # this will need to allow F_k strat as just sum(self.fan_out_list) when we add that 
            draft_lookahead_len = compute_megaspec_lookahead(self.MQ_LEN, self.K)
        elif sync_spec:
            target_lookahead_len = self.K + 1
            draft_lookahead_len = self.K + 1
        else: # draft doesn't matter 
            target_lookahead_len = 1
            draft_lookahead_len = None 

        while self.running and num_seqs_decoded < self.max_num_seqs:
            seq = self.running.popleft()
            # print(f"[scheduler] processing seq {seq.seq_id} for decode, num_tokens={seq.num_tokens}", flush=True)
            
            while not self.bms_can_append(seq, target_lookahead_len, draft_lookahead_len):
                if self.running:  # eject a running sequence if one exists
                    preempted_seq = self.running.pop()
                    self.preempt(preempted_seq)
                else:  # otherwise pop ourselves (ie. current seq)
                    self.preempt(seq) # already popped, will be reinserted at end 
                    break

            else:  # can_append = True and we didn't preempt ourselves, subtle while-else pattern 
                num_seqs_decoded += 1
                self.block_manager.may_append(seq, target_lookahead_len)
                if self.speculate:
                    self.draft_block_manager.may_append(seq, draft_lookahead_len)
                scheduled_seqs.append(seq)

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # print(f"[_preempt] Seq {seq.seq_id}: preempting sequence", flush=True)
        seq.status = SequenceStatus.WAITING
        seq.recovery_token_id = None
        self.block_manager.deallocate(seq)
        if self.speculate:
            self.draft_block_manager.deallocate(seq)
        self.waiting.appendleft(seq) # self.running handled in schedule() when preempt called

        # ── instead, absorb completions as "new prompt" so we re-cache them next prefill
        seq.num_prompt_tokens = seq.num_tokens
        # reinit like it's new, this can be a flag for "am on first spec step"
        seq.last_spec_step_accepted_len = -1

    # non-speculative path, should handle completing a block here as well 
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if is_prefill:
                seq.num_cached_tokens = seq.num_prompt_tokens # no draft needed
            else: 
                seq.num_cached_tokens += 1
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_new_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            else: # if block completes, hash it 
                block_table = seq.block_table
                last_block = self.block_manager.blocks[block_table[-1]]
                
                if seq.last_block_num_tokens == self.block_size:
                    token_ids = seq.block(seq.num_blocks-1)
                    prefix = self.block_manager.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                    h = self.block_manager.compute_hash(token_ids, prefix)
                    # update the last block with the new hash and token ids
                    last_block.update(h, token_ids)
                    self.block_manager.hash_to_block_id[h] = last_block.block_id

    def _handle_eos_and_max_new_tokens(self, seq: Sequence, new_suffix: list[int]) -> list[int]:
        """Handle EOS token detection, max_new_tokens truncation, sequence metadata, and sequence status updates."""
        finished = False

        # Truncate new_suffix at eos if present
        if not seq.ignore_eos and self.eos in new_suffix:
            new_suffix = new_suffix[:new_suffix.index(
                self.eos)+1]  # include eos

        # Truncate new_suffix if it would exceed max_new_tokens
        if seq.num_completion_tokens + len(new_suffix) >= seq.max_new_tokens:
            new_suffix = new_suffix[:seq.max_new_tokens -
                                    seq.num_completion_tokens]

        # Guard against exceeding max_model_len
        if seq.num_tokens + len(new_suffix) > self.max_model_len:
            # Truncate new_suffix to stay within max_model_len
            max_allowed_suffix_len = self.max_model_len - seq.num_tokens
            new_suffix = new_suffix[:max(0, max_allowed_suffix_len)]

        new_suffix_len = len(new_suffix)

        # Check if sequence should be marked as finished
        # Mark as finished if we hit EOS, reach max_new_tokens, max_model_len, or are within speculate_k+1 of max_new_tokens
        if ((not seq.ignore_eos and self.eos in new_suffix) or
                seq.num_completion_tokens + new_suffix_len == seq.max_new_tokens or
                seq.num_tokens + new_suffix_len >= self.max_model_len):  
            finished = True

        assert seq.num_completion_tokens <= seq.max_new_tokens, f"seq.num_completion_tokens = {seq.num_completion_tokens} and seq.max_new_tokens = {seq.max_new_tokens}"

        return new_suffix, finished

    # if finished above, seq.block_table will be [] since it was deallocated()
    def _update_kv_caches(self, seq: Sequence, new_suffix: list[int]):
        """Handle KV cache updates for speculative decoding."""
        # Calculate required blocks after accepting new_suffix
        required_blocks = (seq.num_tokens + len(new_suffix) + self.block_size - 1) // self.block_size
        
        # Calculate what blocks we had allocated for speculation
        spec_blocks_target = len(seq.block_table)
        spec_blocks_draft = len(seq.draft_block_table)
        
        # Determine if we crossed block boundaries during speculation
        spec_crossed_target = spec_blocks_target > required_blocks
        spec_crossed_draft = spec_blocks_draft > required_blocks
        
        # Deallocate excess target blocks if we over-allocated during speculation
        if spec_crossed_target:
            # print(f'spec crossed target', flush=True)
            excess_blocks = spec_blocks_target - required_blocks
            blocks_to_deallocate = seq.block_table[-excess_blocks:]

            for block_id in blocks_to_deallocate:
                block = self.block_manager.blocks[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self.block_manager._deallocate_block(block_id)
            seq.block_table = seq.block_table[:-excess_blocks]
        
        # Deallocate excess draft blocks if we over-allocated during speculation
        if spec_crossed_draft:
            # print(f'spec crossed draft', flush=True)
            excess_blocks = spec_blocks_draft - required_blocks
            blocks_to_deallocate = seq.draft_block_table[-excess_blocks:]
            for block_id in blocks_to_deallocate:
                block = self.draft_block_manager.blocks[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self.draft_block_manager._deallocate_block(block_id)
            seq.draft_block_table = seq.draft_block_table[:-excess_blocks]

    def _finalize_block(self, block_manager, seq: Sequence, block_table: list[int], block_index: int):
        """Finalize a block by computing its hash and updating the cache."""
        token_ids = seq.block(block_index)
        prefix = block_manager.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        h = block_manager.compute_hash(token_ids, prefix)
        last_block = block_manager.blocks[block_table[-1]]
        last_block.update(h, token_ids)
        block_manager.hash_to_block_id[h] = last_block.block_id

    def _update_sequence_metadata(self, seq: Sequence, new_suffix: list[int], recovery_token: int):
        new_suffix_len = len(new_suffix)
        assert new_suffix_len >= 1, "ERROR in _update_sequence_metadata: new_suffix_len = 0, should be non-empty"

        # always need to actually ADD the new suffix to this seq, even after finish
        seq.token_ids.extend(new_suffix)
        seq.num_tokens += new_suffix_len
        seq.last_token = new_suffix[-1]
        seq.num_cached_tokens += new_suffix_len
        seq.num_draft_cached_tokens += new_suffix_len # spec decode touched seqs_copy, now we're updating seqs 
        
        # new recovery token that will be part of next suffix
        seq.last_spec_step_accepted_len = new_suffix_len
        seq.recovery_token_id = recovery_token

        assert seq.last_block_num_tokens == seq.last_block_num_tokens_draft, f"ERROR in _update_sequence_metadata: seq.last_block_num_tokens = {seq.last_block_num_tokens} and seq.last_draft_block_num_tokens = {seq.last_block_num_tokens_draft}"
        assert seq.block_table, "ERROR in _update_sequence_metadata: seq.block_table is empty"
        assert seq.draft_block_table, "ERROR in _update_sequence_metadata: seq.draft_block_table is empty"

        # Finalize all blocks that become complete after accepting new_suffix
        new_total = seq.num_tokens
        for block_index in range(len(seq.block_table)):
            if (block_index + 1) * self.block_size <= new_total:
                # This block is complete
                target_block = self.block_manager.blocks[seq.block_table[block_index]]
                if target_block.hash == -1:
                    self._finalize_block(self.block_manager, seq, seq.block_table, block_index)
                
                draft_block = self.draft_block_manager.blocks[seq.draft_block_table[block_index]]
                if draft_block.hash == -1:
                    self._finalize_block(self.draft_block_manager, seq, seq.draft_block_table, block_index)

    def postprocess_speculate(
        self,
        seqs: list[Sequence],
        new_suffixes: list[list[int]],
        next_recovery_tokens: list[int],
        eagle_acts: torch.Tensor | None = None
    ):

        for i, (seq, new_suffix, next_recovery_token) in enumerate(zip(seqs, new_suffixes, next_recovery_tokens)):
            # ---- EOS/sequence metadata updates (non kv cache metadata) ----
            new_suffix, finished = self._handle_eos_and_max_new_tokens(seq, new_suffix)

            # ---- kv cache updates to roll back to accepted idx (fwd makes kv cache for entire speculation) ----
            self._update_kv_caches(seq, new_suffix)

            # ---- sequence metadata updates ----
            self._update_sequence_metadata(seq, new_suffix, next_recovery_token)

            # ---- EAGLE activation updates for next speculation ----
            if eagle_acts is not None:
                accepted_len = len(new_suffix)
                idx = min(accepted_len - 1, eagle_acts.shape[1] - 1)
                seq.last_target_hidden_state = eagle_acts[i, idx]

            if finished:
                if __debug__: print(f'Sequence {seq.seq_id} finished, deallocating and marking as done + removing from running', flush=True)
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.draft_block_manager.deallocate(seq)
                self.running.remove(seq)
    