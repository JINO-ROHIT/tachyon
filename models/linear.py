# see if we need to consider a seperate file for activations

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, emb_dim: int = 2048, hidden_dim: int = 8192, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, dtype=dtype, bias=False)

    def forward(self, x: torch.Tensor):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2 
        return self.fc3(x)