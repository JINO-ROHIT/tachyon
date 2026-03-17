import torch
import torch.nn as nn

from models.attention import GroupedQueryAttention
from models.linear import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d_in=2048, d_out=2048, num_heads=32, num_kv_groups=8, dtype=torch.bfloat16)
        self.mlp = FeedForward(emb_dim=2048, hidden_dim=8192, dtype=torch.bfloat16)
        self.input_layernorm = nn.RMSNorm(normalized_shape=2048, eps=1e-5, dtype=torch.bfloat16)
        self.post_attention_layernorm = nn.RMSNorm(normalized_shape=2048, eps=1e-5, dtype=torch.bfloat16)

    def forward(self, x, mask, cos, sin):
        shortcut = x
        x = self.input_layernorm(x)
        x = self.att(x, mask, cos, sin)  
        x = x + shortcut 
        
        shortcut = x
        x = self.post_attention_layernorm(x)
        x = self.ff(x)
        x = x + shortcut 
        return x