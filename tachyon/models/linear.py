# see if we need to consider a seperate file for activations

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, emb_dim: int = 2048, hidden_dim: int = 8192, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.up_proj   = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x_fc1 = self.gate_proj(x)
        x_fc2 = self.up_proj(x)
        x = nn.functional.silu(x_fc1) * x_fc2 
        return self.down_proj(x)