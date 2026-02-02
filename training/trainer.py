import torch
from training.loss import language_model_loss
from torch.nn.utils import clip_grad_norm_

class Trainer:
    def __init__(self, model, optimizer, config, device):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            loss = language_model_loss(logits, y)

            self.optimizer.zero_grad()
            loss.backward()

            clip_grad_norm_(
                self.model.parameters(),
                self.config["training"]["grad_clip"]
            )

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)
