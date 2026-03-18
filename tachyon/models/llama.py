import torch
import torch.nn as nn

from .rope import compute_rope_params
from .block import TransformerBlock
from .cache import Cache # lazy import?

class Llama3Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_tokens = nn.Embedding(num_embeddings=128_256, embedding_dim=2048, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(16)])
        self.norm = nn.RMSNorm(normalized_shape=2048, eps=1e-5, dtype=torch.bfloat16)
        self.out_head = nn.Linear(in_features=2048, out_features=128_256, bias=False, dtype=torch.bfloat16)  # this is the weight tied matrix w embed tokens

        cos, sin = compute_rope_params(head_dim=2048 // 32, theta_base=500_000, context_length=131_072)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.current_pos = 0  # track current position in KV cache, better way to do this?

    def forward(self, token_ids, use_cache = True):
        tok_embeds = self.embed_tokens(token_ids)
        x = tok_embeds
        
        num_tokens = x.shape[1]
        
        if use_cache:
            cache = Cache(n_layers=16)
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            mask = torch.triu(torch.ones(pos_end, pos_end, device = x.device, dtype = torch.bool), diagonal = 1)[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0 
            mask = torch.triu(torch.ones(num_tokens, num_tokens, device = x.device, dtype = torch.bool), diagonal = 1)
        
         # (1, 1, num_tokens, num_tokens) to broadcast across batch and heads
        mask = mask[None, None, :, :]

        for i, block in enumerate(self.layers):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos = pos_start,
                                     cache = blk_cache)
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        return logits
