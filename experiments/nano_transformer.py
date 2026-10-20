import torch
import torch.nn as nn

class NanoSelfAttention(nn.Module):
    """A minimal self-attention head from scratch."""
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = torch.softmax(wei, dim=-1)
        
        v = self.value(x)
        out = wei @ v
        return out
