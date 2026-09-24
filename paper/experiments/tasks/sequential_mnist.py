"""
Sequential MNIST task: Treat MNIST images as sequences of pixels.
"""
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class SequentialMNISTDataset(Dataset):
    """
    Converts MNIST images into sequences.
    """

    def __init__(self, mnist_dataset, permute: bool = False, seed: int = 42):
        self.dataset = mnist_dataset
        self.permute = permute
        self.permutation = None
        if permute:
            torch.manual_seed(seed)
            self.permutation = torch.randperm(28 * 28)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # image: (1, 28, 28)
        seq = image.view(-1)  # 784
        if self.permutation is not None:
            seq = seq[self.permutation]
        return seq, label


def create_sequential_mnist_dataloaders(
    data_dir: str,
    batch_size: int,
    permute: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train/val dataloaders for Sequential MNIST.
    """
    data_dir = Path(data_dir)
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_seq = SequentialMNISTDataset(train_dataset, permute=permute, seed=42)
    val_seq = SequentialMNISTDataset(val_dataset, permute=permute, seed=123)

    train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_seq, batch_size=batch_size, shuffle=False)

    input_dim = 28 * 28
    return train_loader, val_loader, input_dim
