import json
from pathlib import Path
import torch

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.stoi, f)

    @staticmethod
    def load(path: Path):
        with open(path) as f:
            stoi = json.load(f)
        tokenizer = CharTokenizer("")
        tokenizer.stoi = stoi
        tokenizer.itos = {i: c for c, i in stoi.items()}
        tokenizer.vocab_size = len(stoi)
        return tokenizer


def prepare_dataset(raw_path, processed_dir):
    text = Path(raw_path).read_text(
        encoding="utf-8",
        errors="replace"
    )

    tokenizer = CharTokenizer(text)
    encoded = tokenizer.encode(text)

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save(processed_dir / "vocab.json")
    torch.save(encoded, processed_dir / "data.pt")

    print(f"Saved vocab size: {tokenizer.vocab_size}")
    print(f"Total tokens: {len(encoded)}")

if __name__ == "__main__":
    prepare_dataset(
        raw_path="data/raw/blogs.txt",
        processed_dir="data/processed"
    )
