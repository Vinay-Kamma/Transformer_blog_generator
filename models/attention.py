import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # One projection per Q, K, V (projecting into all heads at once)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection (mix heads)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        return: (B, T, D)
        """
        B, T, D = x.shape

        # 1️⃣ Linear projections
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2️⃣ Split into heads
        # (B, T, D) → (B, H, T, d_k)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # 3️⃣ Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (B, H, T, T)

        # 4️⃣ Causal mask (no future tokens)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # 5️⃣ Softmax → attention weights
        attn = torch.softmax(scores, dim=-1)

        # 6️⃣ Weighted sum of values
        out = attn @ V  # (B, H, T, d_k)

        # 7️⃣ Combine heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # 8️⃣ Final projection
        return self.W_o(out)
