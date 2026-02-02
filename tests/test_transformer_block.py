import torch
from models.transformer_block import TransformerBlock

def test_transformer_block_shape():
    B, T, D = 2, 16, 64
    H = 4

    block = TransformerBlock(d_model=D, num_heads=H)
    x = torch.randn(B, T, D)

    out = block(x)

    assert out.shape == (B, T, D)
