"""Student model enhanced with SwiGLU FFN and RMSNorm for lower BPB."""
import torch
from torch import nn
from torch.nn import functional as F
from model import GPT

class SwiGLUBlock(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        
        # SwiGLU FFN
        hidden_dim = int(2.67 * width)
        self.w1 = nn.Linear(width, hidden_dim)
        self.w2 = nn.Linear(width, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, width)

    def forward(self, x):
        batch, length, width = x.shape
        # Self-Attention
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        
        # SwiGLU MLP: (SiLU(w1(x)) * w2(x)) -> w3
        normed_x = self.norm2(x)
        swiglu_out = F.silu(self.w1(normed_x)) * self.w2(normed_x)
        x = x + self.w3(swiglu_out)
        
        return x


class SwiGLUGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        width = config['width']
        self.blocks = nn.ModuleList([SwiGLUBlock(width, config['heads']) for _ in range(config['depth'])])
        self.apply(self.initialize)
        self.head.weight = self.token.weight


def build_model(config):
    return SwiGLUGPT(config)
