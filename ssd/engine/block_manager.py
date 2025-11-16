from collections import deque
import xxhash
import numpy as np

from ssd.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(
        self, 
        num_blocks: int, 
        block_size: int, 
        is_draft: bool = False, 
        speculate_k: int = -1, 
        max_model_len: int = -1,
        verbose: bool = False
    ):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.is_draft: bool = is_draft
        self.speculate_k: int = speculate_k 
        self.verbose: bool = verbose
        self.max_model_len: int = max_model_len

        
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _allocate_n_blocks(self, n: int) -> list[Block]:
        if len(self.free_block_ids) < n:
            raise RuntimeError(f"Insufficient free blocks: need {n}, have {len(self.free_block_ids)}")
        
        # Extract n block IDs in one operation
        block_ids = [self.free_block_ids.popleft() for _ in range(n)]
        # block_ids = [self.free_block_ids.pop() for _ in range(n)] # does avoiding recently used blocks fix the allocation of [0, 1] to second seq? wezdoo 

        # Reset all blocks and update tracking sets
        blocks = []
        for block_id in block_ids:
            block = self.blocks[block_id]
            assert block.ref_count == 0
            block.reset()
            self.used_block_ids.add(block_id)
            blocks.append(block)
        
        return blocks

    def _deallocate_n_blocks(self, block_ids: list[int]): # we need to separate where we do ref_count -=1 for forked things from the act of deallocation itself 
        for block_id in block_ids:
            block = self.blocks[block_id]
            block.ref_count -= 1 # added this -- keeping the assert ensures we only have our fork clones pointing to each seq
            if block.ref_count == 0: 
                self._deallocate_block(block_id)


    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        
        if self.blocks[block_id].hash != -1: # if block was finalized, remove from hash_to_block_id checkme
            if self.hash_to_block_id.get(self.blocks[block_id].hash) == block_id:
                del self.hash_to_block_id[self.blocks[block_id].hash]

        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        block_table = seq.draft_block_table if self.is_draft else seq.block_table
        assert not block_table 
        h = -1 
        cache_miss = False

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else: # cache hit 
                if self.is_draft: 
                    seq.num_draft_cached_tokens += self.block_size
                else:
                    seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            block_table.append(block_id)


    def deallocate(self, seq: Sequence):
        block_table = seq.draft_block_table if self.is_draft else seq.block_table
        for block_id in reversed(block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        if self.is_draft:
            seq.num_draft_cached_tokens = 0
        else: 
            seq.num_cached_tokens = 0

        block_table.clear()

    def can_append(self, seq: Sequence, lookahead_num_tokens: int = 1) -> bool:
        block_table = seq.draft_block_table if self.is_draft else seq.block_table

        # Check if sequence length + lookahead would exceed max model length
        if seq.num_tokens + lookahead_num_tokens > self.max_model_len:
            print(f'[block_manager] WARNING: Sequence length + lookahead would exceed max model length', flush=True)
            return False

        # How many blocks do we need in total to cover current tokens + lookahead?
        target_blocks = (seq.num_tokens + lookahead_num_tokens +
                         self.block_size - 1) // self.block_size
        current_blocks = len(block_table)

        if target_blocks > current_blocks:
            needed = target_blocks - current_blocks
            return len(self.free_block_ids) >= needed
        else:
            return True  # Current blocks are sufficient

    def may_append(self, seq: Sequence, lookahead_num_tokens: int = 1):
        block_table = seq.draft_block_table if self.is_draft else seq.block_table

        # How many blocks do we need in total to cover current tokens + lookahead?
        target_blocks = (seq.num_tokens + lookahead_num_tokens +
                         self.block_size - 1) // self.block_size
        current_blocks = len(block_table)

        if target_blocks > current_blocks:
            needed = target_blocks - current_blocks
            new_blocks = self._allocate_n_blocks(needed)
            for block in new_blocks:
                block_table.append(block.block_id)

