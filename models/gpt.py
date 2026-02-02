import math
import torch
import torch.nn as nn

from models.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int
    ):
        super().__init__()

        # Token + positional embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        # Final normalization
        self.ln_f = nn.LayerNorm(d_model)

        # Language modeling head (weight tied)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

        self.d_model = d_model
        self.max_len = max_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T)
        returns logits: (B, T, V)
        """
        B, T = idx.shape
        assert T <= self.max_len, "Sequence length exceeds model context window"

        # Embedding + scaling
        x = self.token_embed(idx) * math.sqrt(self.d_model)

        # Positional encoding
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = x + self.pos_embed(pos)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Output logits
        logits = self.lm_head(x)
        return logits
