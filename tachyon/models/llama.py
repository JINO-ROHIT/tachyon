from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .rope import compute_rope_params
from .block import TransformerBlock

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

    def forward(self, token_ids, caches: List[Optional["Cache"]], start_positions: List[int]):
        batch_size, num_tokens = token_ids.shape

        x = self.embed_tokens(token_ids) # here x becomes (batch_size, tokens, embed_dim)
        
        max_seq = max(start_positions) + num_tokens
        full_mask = torch.triu(torch.ones(max_seq, max_seq, device=x.device, dtype=torch.bool), diagonal=1)

        mask_rows = [] # we need to create masks relative to the largest size max_seq
        for i, sp in enumerate(start_positions):
            row = full_mask[sp : sp + num_tokens, :max_seq]   # (T, max_seq)
            mask_rows.append(row)
        mask = torch.stack(mask_rows, dim=0).unsqueeze(1)

        for layer_idx, block in enumerate(self.layers):
            layer_caches = [(c.get(layer_idx) if c is not None else None) for c in caches]
            x, new_layer_caches = block(x, mask, self.cos, self.sin, start_positions, layer_caches)

            for i, c in enumerate(caches):
                if c is not None:
                    c.update(layer_idx, new_layer_caches[i])
 
        x = self.norm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        return logits
