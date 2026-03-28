import torch
from typing import List, Union
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from queue import Queue

from tachyon.models import Llama3Model
from tachyon.utils import load_weights
from tachyon.models.cache import Cache  # lazy import?
from tachyon.models.paged_attention import Request

device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_BATCH_SIZE = 50  # TO-DO: write a profiler to find optimal size that maximizes gpu util


# @dataclass
# class Request:
#     id: int = 0
#     prompt: str = ""
#     max_tokens: int = 100
#     temperature: float = 0.1

#     prompt_tokens: List[int] = field(default_factory=list)
#     tokens: List[int] = field(default_factory=list)    
#     kv_cache: Cache | None = field(default_factory=lambda: Cache(n_layers=16))

#     cache_pos: int = 0
#     is_completed: bool = False
#     is_prefill: bool = True
#     use_cache: bool = True

#     response: str = None # the actual string decoded response


class Engine:
    def __init__(self, model_name: str):
        self.model = Llama3Model()  # for now we only have llama
        load_weights(self.model, model_name.split("/")[-1])  # loads in place
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        self.pool = Queue()
        self.current_batch = []

        self.request_id = 0

    def add_request(self, request: Request):
        """adds a request in the pool"""
        self.pool.put(request)

    def sample(self, logits, temperature: float = 0.0):
        """samples from a distribution to get the next token"""
        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token

    def prefill(self, request: Request):
        """the prefill stage"""
        request.cache_pos = 0
        prompt_tokens = self.tokenizer.encode(request.prompt)
        tokens = torch.tensor(prompt_tokens).unsqueeze(0).to(device)
        request.prompt_tokens = prompt_tokens
        if not request.use_cache:
            request.kv_cache = None

        with torch.no_grad():
            logits = self.model(tokens, caches=[request.kv_cache], start_positions=[0])

        next_token = self.sample(logits[:, -1, :], request.temperature)
        request.tokens.append(next_token.item())

        request.cache_pos = len(prompt_tokens)
        request.is_prefill = False
    
    def decode_batch(self, requests: List[Request]):
        """the decode step for a batch of requests in a single forward pass."""
        # stack the last token of each request: (B, 1)
        tokens = torch.tensor([[r.tokens[-1]] for r in requests], device=device)
        caches = [r.kv_cache for r in requests]
        start_positions = [r.cache_pos for r in requests]
 
        with torch.no_grad():
            logits = self.model(tokens, caches=caches, start_positions=start_positions)
        last_logits = logits[:, -1, :]
 
        next_tokens = torch.stack([self.sample(last_logits[i:i+1], requests[i].temperature).squeeze() for i in range(len(requests))]) 
 
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
 
        # prefill individually 
        for request in prefill_requests:
            self.prefill(request)
 
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