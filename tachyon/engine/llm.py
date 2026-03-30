import torch
from typing import List, Union
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from queue import Queue

from tachyon.models import Llama3Model
from tachyon.utils import load_weights
from tachyon.models.cache import Cache, PrefixCache

device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_BATCH_SIZE = 50  # TO-DO: write a profiler to find optimal size that maximizes gpu util
MAX_DECODE_BATCH = 50
MAX_PREFILL_BATCH = 30

@dataclass
class Request:
    id: int = 0
    prompt: str = ""
    max_tokens: int = 100
    temperature: float = 0.1

    prompt_tokens: List[int] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)    
    kv_cache: Cache | None = field(default_factory=lambda: Cache(n_layers=16))

    cache_pos: int = 0 # keeps track of upto what token position has already been processed
    is_completed: bool = False
    is_prefill: bool = True
    use_cache: bool = True # to use kv cache or not

    response: str = None # the actual string decoded response


class Engine:
    def __init__(self, model_name: str):
        self.model = Llama3Model()  # for now we only have llama
        load_weights(self.model, model_name.split("/")[-1])  # loads in place
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        self.pool = Queue()
        self.current_batch = []
        self.prefix_cache = PrefixCache()

        self.request_id = 0

    def add_request(self, request: Request):
        """adds a request in the pool"""
        self.pool.put(request)

    def sample(self, logits, temperatures: torch.Tensor):
        """
        logits: (B, vocab_size)
        temperatures: (B, 1)
        """
        temperatures = temperatures.to(dtype=logits.dtype, device=logits.device)
        mask = temperatures.squeeze(-1) > 0.0

        scaled = logits.clone()
        scaled[mask] = scaled[mask] / temperatures[mask]
        scaled[mask] = scaled[mask] - scaled[mask].max(dim=-1, keepdim=True).values
        probs = torch.softmax(scaled, dim=-1)  # (B, vocab)

        # greedy for zero-temp rows
        greedy = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)

        # sample for nonzero-temp rows
        sampled = torch.multinomial(probs, num_samples=1)    # (B, 1)

        next_tokens = torch.where(mask.unsqueeze(-1), sampled, greedy)
        return next_tokens  # (B, 1)
    
    def _forward_pass(self, tokens, caches, cache_positions):
        """does one forward pass of the model and returns the next token"""
        with torch.no_grad():
            logits = self.model(tokens, caches=caches, start_positions=cache_positions)
        return logits[:, -1, :]

    def prefill_batch(self, requests: List[Request]):
        """batched prefill for multiple requests in a single forward pass"""

        all_tokens = [self.tokenizer.encode(r.prompt) for r in requests]
        max_len = max(len(t) for t in all_tokens)

        # pad all to max_len (pad on the left so the last token is always real)
        pad_id = self.tokenizer.pad_token_id or 0
        padded = [([pad_id] * (max_len - len(t))) + t for t in all_tokens]
        tokens = torch.tensor(padded, device=device)  # (B, max_len)

        # start positions are all 0 since these are fresh requests
        caches = [r.kv_cache if r.use_cache else None for r in requests]
        start_positions = [0] * len(requests)

        logits = self._forward_pass(tokens, caches, start_positions)  # (B, vocab)
        temps = torch.tensor([[r.temperature] for r in requests], device=device)
        next_tokens = self.sample(logits, temps)  # (B, 1)

        for i, request in enumerate(requests):
            request.prompt_tokens = all_tokens[i]
            request.tokens.append(next_tokens[i].item())
            request.cache_pos = max_len  # all requests share the same padded length
            request.is_prefill = False

            if request.use_cache:
                self.prefix_cache.store(all_tokens[i], [request.kv_cache.get(j) for j in range(16)])

    def _prefill_with_cache(self, request: Request):
        """prefill a single request that has a prefix cache hit, skipping redundant prefix computation"""
        prompt_tokens = request.prompt_tokens
        match_len, cached_kv = self.prefix_cache.lookup(prompt_tokens)

        # for exact match we trim last position so we can recompute it for logits
        use_len = min(match_len, len(prompt_tokens) - 1)
        for layer_idx, (k, v) in enumerate(cached_kv):
            request.kv_cache.update(layer_idx, (k[:, :, :use_len, :].clone(), v[:, :, :use_len, :].clone()))

        remaining = torch.tensor(prompt_tokens[use_len:]).unsqueeze(0).to(device)
        logits = self._forward_pass(remaining, [request.kv_cache], [use_len])

        temps = torch.tensor([[request.temperature]], device=device)
        next_token = self.sample(logits, temps)
        request.tokens.append(next_token.item())

        request.cache_pos = len(prompt_tokens)
        request.is_prefill = False

        self.prefix_cache.store(prompt_tokens, [request.kv_cache.get(i) for i in range(16)])
    
    def decode_batch(self, requests: List[Request]):
        """the decode step for a batch of requests in a single forward pass."""
        # stack the last token of each request: (B, 1)
        tokens = torch.tensor([[r.tokens[-1]] for r in requests], device=device)
        caches = [r.kv_cache for r in requests]
        start_positions = [r.cache_pos for r in requests]

        logits = self._forward_pass(tokens, caches, start_positions)
        temps = torch.tensor([[r.temperature] for r in requests], device=device)  # (B, 1)
        next_tokens = self.sample(logits, temps) 

        for i, request in enumerate(requests):
            request.cache_pos += 1
            tok = next_tokens[i].item()
            request.tokens.append(tok)
            if tok == self.tokenizer.eos_token_id or len(request.tokens) >= request.max_tokens:
                request.is_completed = True
                request.response = self.tokenizer.decode(request.tokens)

    def _get_next_batch(self):
        self.current_batch = [_req for _req in self.current_batch if not _req.is_completed]  # in the current batch keep the ones not completed

        # now to fill the remaining gap, add many new requests from the pool
        while not self.pool.empty() and len(self.current_batch) < MAX_BATCH_SIZE:
            self.current_batch.append(self.pool.get())

        return self.current_batch

    def generate(self):
        self.current_batch = self._get_next_batch()
        if not self.current_batch:
            return False
 
        prefill_requests = [r for r in self.current_batch if r.is_prefill]
        decode_requests  = [r for r in self.current_batch if not r.is_prefill]

        # split prefill into cache hits (handled individually) and misses (batched)
        cache_hits, cache_misses = [], []
        for request in prefill_requests:
            if request.use_cache:
                prompt_tokens = self.tokenizer.encode(request.prompt)
                request.prompt_tokens = prompt_tokens  
                match_len, _ = self.prefix_cache.lookup(prompt_tokens)
                match_ratio = match_len / len(prompt_tokens)
                # why ratio? if we use a len based matching, even if a single token match, we use prefill cache but prefill cache has no batching, so we lose 
                # throughput
                if match_ratio > 0.3: # tune this 
                    cache_hits.append(request)
                    continue
            cache_misses.append(request)

        for request in cache_hits:
            self._prefill_with_cache(request)

        # prefill misses in smaller batches to not OOM the GPU
        for i in range(0, len(cache_misses), MAX_PREFILL_BATCH):
            chunk = cache_misses[i : i + MAX_PREFILL_BATCH]
            self.prefill_batch(chunk)

        # decode in one batched forward pass
        if decode_requests:
            self.decode_batch(decode_requests)
 
        return True
    
    def generate_text(self, prompts: Union[str, List[str]], max_tokens: int = 100, temperature: float = 0.1):
        """the public api for generation"""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        requests = []
        for prompt in prompts:
            req = Request(id=self.request_id, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            self.request_id += 1
            self.add_request(req)
            requests.append(req)
        
        # run engine loop
        while self.generate():
            pass

        return requests[0] if len(requests) == 1 else requests
        

if __name__ == "__main__":
    engine = Engine("meta-llama/Llama-3.2-1B-Instruct")
    print(engine.generate_text("Explain AGI"))

    # outputs = engine.generate_text([
    #     "Explain AGI",
    #     "What is vLLM?",
    #     "Tell me about SGLang"
    # ])

    # for o in outputs:
    #     print(o)