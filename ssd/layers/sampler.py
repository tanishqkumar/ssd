import torch
from torch import nn

from ssd.utils.async_helpers.async_spec_helpers import apply_sampler_x_rescaling

torch.manual_seed(0) 

class Sampler(nn.Module): 
    def __init__(self, sampler_x: float | None = None, async_fan_out: int = 3):
        super().__init__()
        self.sampler_x = sampler_x
        self.F = async_fan_out # will need to accomodate lists for hit/miss eventually 
    
    @torch.inference_mode() # what shape are logits during tree decode? MQ_LEN, 
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, is_tree: bool = False):
        # logits: [B, V], temperatures: [B]
        
        logits_cpy = logits.to(torch.float) 
        greedy_tokens = logits_cpy.argmax(dim=-1)

        # Fast path: any zero temperature rows are greedy
        temps = temperatures
        zero_mask = temps == 0
        
        # Note: keep inplace ops for speed
        logits_cpy.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits_cpy, dim=-1, dtype=torch.float)
        
        # Apply sampler_x rescaling when conditions are met
        if self.sampler_x is not None and is_tree:
            probs = apply_sampler_x_rescaling(probs, self.sampler_x, self.F)
        
        epsilon = 1e-10
        scores = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon)
        sample_tokens = scores.argmax(dim=-1)
        return torch.where(zero_mask, greedy_tokens, sample_tokens)


def profile_sampler():
    """Profile the sampler on [b, v] logits for b=128, v=150_000"""
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nProfiling Sampler on {device}")
    
    # Test parameters
    b = 128
    v = 150_000
    
    # Create test data
    logits = torch.randn(b, v, device=device)
    temperatures = torch.rand(b, device=device) * 1.5  # temperatures in [0, 1.5]
    
    sampler = Sampler().to(device)
    
    print(f"Testing with batch_size={b}, vocab_size={v}")
    
    # Warm up
    print("Warming up sampler")
    for _ in range(10):
        _ = sampler(logits, temperatures)
    
    # Profile
    num_runs = 100
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        _ = sampler(logits, temperatures)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    sampler_time_ms = (end_time - start_time) * 1000 / num_runs
    
    print(f"Sampler time: {sampler_time_ms:.3f}ms")

# takes 0.5ms, negligible 
if __name__ == "__main__":
    profile_sampler()


''' 
Fancy = False (greedy, mid)

    python bench.py --size 1 --draft 1 --llama --spec --k 6 --async --gpus 2 --numseqs 1 --b 1 --f 3 --ttemp 1.0
        [metrics] Avg Tokens per step (incl recovery): 3.78
        [metrics] Avg Fraction of Speculated Tokens Accepted: 0.46
        [metrics] Avg target time per full step (ms): 262.44
        [metrics] Avg Cache Hits: 0.79
        [metrics] Avg Tokens per step on Cache Hit: 4.50
        [metrics] Empirical frequencies of accepted_suffix_lens_on_hit - 1:
        0: 0.213
        1: 0.148
        2: 0.056
        3: 0.046
        4: 0.037
        5: 0.046
        6: 0.454
        Model: Llama-3.2-1B-Instruct, Mode: CUDA Graphs + Speculative(k=6) + Async, Total: 512tok, Time: 35.96s, Total Throughput: 14.24tok/s


Fancy = True (best)

    python bench.py --size 1 --draft 1 --llama --spec --k 6 --async --gpus 2 --numseqs 1 --b 1 --f 3 --ttemp 1.0 --dtemp 1.0  --fancy 
    
    Final Prefill Throughput: 46tok/s
Final Decode Throughput: 23tok/s
[metrics] Avg Tokens per step (incl recovery): 6.08
[metrics] Avg Fraction of Speculated Tokens Accepted: 0.85
[metrics] Avg target time per full step (ms): 266.18
[metrics] Avg Cache Hits: 0.87
[metrics] Avg Tokens per step on Cache Hit: 6.84
[metrics] Empirical frequencies of accepted_suffix_lens_on_hit - 1:
  0: 0.000
  1: 0.027
  2: 0.000
  3: 0.000
  4: 0.014
  5: 0.000
  6: 0.959
Model: Llama-3.2-1B-Instruct, Mode: CUDA Graphs + Speculative(k=6) + Async + Fancy, Total: 512tok, Time: 22.89s, Total Throughput: 22.37tok/s


False = False, temps equal (worst)


--- 

without is_tree but with fancy 

Final Prefill Throughput: 66tok/s
Final Decode Throughput: 4tok/s
[metrics] Avg Tokens per step (incl recovery): 1.14
[metrics] Avg Fraction of Speculated Tokens Accepted: 0.02
[metrics] Avg target time per full step (ms): 261.74
[metrics] Avg Cache Hits: 0.13
[metrics] Avg Tokens per step on Cache Hit: 2.11
[metrics] Empirical frequencies of accepted_suffix_lens_on_hit - 1:
  0: 0.487
  1: 0.292
  2: 0.071
  3: 0.053
  4: 0.027
  5: 0.018
  6: 0.053




''' 