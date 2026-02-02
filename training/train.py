import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from data.dataset import CharDataset
from models.gpt import GPT
from training.trainer import Trainer
from training.checkpoint import save_checkpoint


def main():
    config = yaml.safe_load(open("config.yaml"))

    device = torch.device(config["training"]["device"])

    # Load processed data
    data = torch.load("data/processed/data.pt")
    dataset = CharDataset(data, config["data"]["block_size"])
    loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True
    )

    # Model
    model = GPT(
        vocab_size=len(set(data)),
        max_len=config["model"]["max_len"],
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )

    trainer = Trainer(model, optimizer, config, device)

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        loss = trainer.train_epoch(loader)
        print(f"Epoch {epoch}: loss = {loss:.4f}")

        save_checkpoint(
            model,
            optimizer,
            epoch,
            Path("checkpoints") / f"epoch_{epoch}.pt"
        )

if __name__ == "__main__":
    main()
