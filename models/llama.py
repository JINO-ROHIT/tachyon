import torch
import torch.nn as nn

from models.rope import compute_rope_params
from models.block import TransformerBlock

class Llama3Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_tokens = nn.Embedding(num_embeddings=128_256, embedding_dim=2048, dtype=torch.bfloat16)
        self.trf_blocks = nn.ModuleList([TransformerBlock() for _ in range(16)])
        self.post_attention_layernorm = nn.RMSNorm(normalized_shape=2048, eps=1e-5, dtype=torch.bfloat16)
        self.out_head = nn.Linear(in_features=2048, out_features=128_256, bias=False, dtype=torch.bfloat16) # this is the weight tied matrix w embed tokens

        cos, sin = compute_rope_params(head_dim=2048 // 32, theta_base=500_000, context_length=131_072,)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)


    def forward(self, in_idx):

        tok_embeds = self.embed_tokens(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.post_attention_layernorm(x)
        logits = self.out_head(x.to(torch.bfloat16))
        return logits