import torch

def compute_rope_params(head_dim: int, theta_base: int=10_000, context_length: int=4096, dtype: torch.dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) 
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles) # (context_len, head_dim)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x, cos, sin, offset = 0):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., : head_dim // 2]  # first half
    x2 = x[..., head_dim // 2 :]  # second half

    cos = cos[offset: offset + seq_len, :].unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, head_dim)
    sin = sin[offset: offset + seq_len, :].unsqueeze(0).unsqueeze(0) 


    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)

def apply_rope_vectorized(x, cos, sin, start_positions = 0):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., : head_dim // 2]  # first half
    x2 = x[..., head_dim // 2 :]  # second half

    positions = torch.arange(seq_len, device = x.device).unsqueeze(0) # (1, seq_len)
    absolute_positions = positions + torch.tensor(start_positions, device=x.device).unsqueeze(1) # (batch_size, seq_len)

    cos_selected = cos[absolute_positions]  # (batch, seq_len, head_dim)
    sin_selected = sin[absolute_positions]  # (batch, seq_len, head_dim)
    
    cos_selected = cos_selected.unsqueeze(1)  
    sin_selected = sin_selected.unsqueeze(1)  
    
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos_selected) + (rotated * sin_selected)
    
    return x_rotated.to(dtype=x.dtype)