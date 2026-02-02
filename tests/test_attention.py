import torch
from models.attention import MultiHeadSelfAttention

def test_attention_forward_shape():
    B, T, D = 2, 16, 64
    H = 4

    attn = MultiHeadSelfAttention(d_model=D, num_heads=H)
    x = torch.randn(B, T, D)

    out = attn(x)

    assert out.shape == (B, T, D)
