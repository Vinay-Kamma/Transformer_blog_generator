import torch
import torch.nn as nn
from models.attention import MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        # Pre-LN design
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        # 1️⃣ Attention with residual
        x = x + self.attn(self.ln1(x))

        # 2️⃣ Feed-forward with residual
        x = x + self.ffn(self.ln2(x))

        return x
