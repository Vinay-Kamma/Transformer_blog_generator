import torch
import json
from models.gpt import GPT
from inference.generate import generate


def load_tokenizer(path):
    with open(path) as f:
        stoi = json.load(f)
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def generate_blog(
    checkpoint_path,
    vocab_path,
    prompt,
    max_tokens=300,
    temperature=0.8,
    device="cpu"
):
    stoi, itos = load_tokenizer(vocab_path)

    model = GPT(
        vocab_size=len(stoi),
        max_len=128,
        d_model=256,
        num_heads=8,
        num_layers=6
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    start_ids = torch.tensor([[stoi[c] for c in prompt]], device=device)

    out = generate(model, start_ids, max_tokens, temperature)
    text = "".join(itos[i.item()] for i in out[0])

    return text
