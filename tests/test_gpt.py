import torch
from models.gpt import GPT

def test_gpt_forward_shape():
    B, T = 2, 16
    V = 50
    D = 64
    H = 4
    L = 3

    model = GPT(
        vocab_size=V,
        max_len=T,
        d_model=D,
        num_heads=H,
        num_layers=L
    )

    x = torch.randint(0, V, (B, T))
    logits = model(x)

    assert logits.shape == (B, T, V)
