"""
Character-level language modeling task (enwik8).
"""
import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
ENWIK8_MD5 = "a1fa5ffddb56f4953e226637dabbb36a"


def _download_enwik8(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "enwik8.zip"
    data_path = data_dir / "enwik8"

    if data_path.exists():
        return data_path

    if not zip_path.exists():
        print(f"Downloading enwik8 from {ENWIK8_URL}...")
        urllib.request.urlretrieve(ENWIK8_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    return data_path


class CharLMDataset(Dataset):
    """
    Character-level language modeling dataset.

    Input: sequence of length N
    Target: same sequence shifted by 1
    """

    def __init__(self, data: bytes, seq_len: int):
        self.data = np.frombuffer(data, dtype=np.uint8)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len].astype(np.int64)
        y = self.data[idx + 1 : idx + self.seq_len + 1].astype(np.int64)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def create_language_model_dataloaders(
    data_dir: str,
    seq_len: int,
    batch_size: int,
    train_fraction: float = 0.9,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train/val dataloaders for enwik8 character-level LM.
    """
    data_dir = Path(data_dir)
    data_path = _download_enwik8(data_dir)

    with open(data_path, "rb") as f:
        data = f.read()

    split_idx = int(len(data) * train_fraction)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_dataset = CharLMDataset(train_data, seq_len)
    val_dataset = CharLMDataset(val_data, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    vocab_size = 256  # byte-level vocabulary
    return train_loader, val_loader, vocab_size
