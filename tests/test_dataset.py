import torch
from data.dataset import CharDataset

def test_dataset_shift():
    data = [1, 2, 3, 4, 5]
    block_size = 3
    ds = CharDataset(data, block_size)

    x, y = ds[0]

    assert torch.equal(x, torch.tensor([1, 2, 3]))
    assert torch.equal(y, torch.tensor([2, 3, 4]))
