from typing import List, Optional

import torch
import torch.nn as nn

from .rope import apply_rope, apply_rope_vectorized
from .paged_attention import Page, PageTable, BlockManager
from tachyon.utils import pad_to

'''
some mental model for this - we have many Q heads but limited k/v heads that share it, 
the idea is to use lesser kv heads to optimize it and then match the shape.

num_heads = num of query heads
num_kv_groups = num of k/v heads

num_heads = 4
num_kv_groups = 2

Q1 = K1 V1
Q2 = K1 V1   (shared)

Q3 = K2 V2
Q4 = K2 V2   (shared)
'''

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups"
        )

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.k_proj = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.q_proj = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_positions: List[int], caches: List[Optional["Cache"]]):
        batch_size, num_tokens, _ = x.shape # (batch, tokens, embed_dim)

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # apply per request (i:i+1) trick is to keep the dim, if we do i, we lose the dim, see if we can vectorize this instead of for loop
        keys = torch.stack([apply_rope(keys[i: i + 1], cos, sin, start_positions[i]) for i in range(batch_size)], dim = 0).squeeze(1)
        queries = torch.stack([apply_rope(queries[i : i + 1], cos, sin, start_positions[i]) for i in range(batch_size)], dim = 0).squeeze(1)

        # keys = apply_rope_vectorized(keys, cos, sin, start_positions)
        # queries = apply_rope_vectorized(queries, cos, sin, start_positions)

        next_caches = []
        keys_list, values_list = [], []
        for i in range(batch_size):
            k_i, v_i = keys[i:i+1], values[i:i+1]   # (1, num_kv_groups, tokens, head_dim)
            if caches[i] is not None:
                prev_k, prev_v = caches[i]
                k_i = torch.cat([prev_k, k_i], dim=2)
                v_i = torch.cat([prev_v, v_i], dim=2)
            next_caches.append((k_i, v_i))
            keys_list.append(k_i)
            values_list.append(v_i)
        
        # pad all kv tensor to same seq length in the batch for attention to work
        max_seq = max(k.shape[2] for k in keys_list)

        keys_padded   = torch.cat([pad_to(k, max_seq) for k in keys_list],   dim=0)  # (B, kv_group, max_seq, D)
        values_padded = torch.cat([pad_to(v, max_seq) for v in values_list], dim=0)
 
        keys_padded   = keys_padded.repeat_interleave(self.group_size, dim=1)   # (B, H, max_seq, D)
        values_padded = values_padded.repeat_interleave(self.group_size, dim=1)
 
        attn_scores = queries @ keys_padded.transpose(2, 3)           # (B, H, T, max_seq)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1)
 
        context_vec = (attn_weights @ values_padded).transpose(1, 2)  # (B, T, H, D)
        context_vec = context_vec.reshape(batch_size, num_tokens, self.d_out)
        return self.o_proj(context_vec), next_caches

        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]