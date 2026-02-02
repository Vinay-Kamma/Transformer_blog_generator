import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, data, block_size: int):
        """
        data: 1D list or tensor of token IDs
        block_size: sequence length T
        """
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        # last block has no target
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y
