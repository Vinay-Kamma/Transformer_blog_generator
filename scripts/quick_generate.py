import torch
import yaml
import json
from pathlib import Path

from models.gpt import GPT
from inference.generate import generate


def main():
    # Load config (single source of truth)
    config = yaml.safe_load(open("config.yaml"))

    device = torch.device("cpu")

    # Load vocab
    vocab_path = Path("data/processed/vocab.json")
    assert vocab_path.exists(), "Vocab file not found"

    stoi = json.load(open(vocab_path))
    itos = {i: c for c, i in stoi.items()}

    # Load checkpoint
    ckpt_path = Path("checkpoints/epoch_4.pt")
    assert ckpt_path.exists(), "Checkpoint not found"

    ckpt = torch.load(ckpt_path, map_location=device)

    # Build model EXACTLY as trained
    model = GPT(
        vocab_size=len(stoi),
        max_len=config["model"]["max_len"],
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
    ).to(device)

    model.load_state_dict(ckpt["model_state"])

    # Prompt (short, safe)
    prompt = "Today "
    start_ids = torch.tensor([[stoi[c] for c in prompt]], device=device)

    # Generate very few tokens
    out = generate(
        model,
        start_ids,
        max_new_tokens=40,
        temperature=0.8
    )

    text = "".join(itos[i.item()] for i in out[0])

    # Hard sanity check
    assert len(text) > len(prompt), "Generation failed"

    print("Quick generation success:")
    print(text)


if __name__ == "__main__":
    main()
