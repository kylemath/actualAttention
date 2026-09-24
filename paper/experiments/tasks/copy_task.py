"""
Copy task: Test ability to memorize and reproduce a sequence.
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CopyTaskDataset(Dataset):
    """
    Copy task dataset.
    
    Input: [sequence] [delimiter] [blanks]
    Target: [blanks] [sequence]
    
    Tests ability to store and retrieve information.
    """
    
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int,
                 blank_len: int = 10, seed: int = 42):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.blank_len = blank_len
        
        np.random.seed(seed)
        
        # Generate all sequences
        self.inputs = []
        self.targets = []
        
        # Reserve special tokens
        self.blank_token = 0
        self.delimiter_token = vocab_size - 1
        
        for _ in range(num_samples):
            # Random sequence (excluding special tokens)
            sequence = np.random.randint(1, vocab_size - 1, size=seq_len)
            
            # Input: [sequence] [delimiter] [blanks]
            input_seq = np.concatenate([
                sequence,
                [self.delimiter_token],
                np.full(blank_len, self.blank_token)
            ])
            
            # Target: [blanks after delimiter] [sequence]
            # Length must match input length (seq_len + 1 + blank_len)
            target_seq = np.concatenate([
                np.full(blank_len + 1, self.blank_token),
                sequence
            ])
            
            self.inputs.append(input_seq)
            self.targets.append(target_seq)
        
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


def create_copy_task_dataloaders(num_train: int, num_val: int, seq_len: int,
                                   vocab_size: int, batch_size: int, blank_len: int = 10):
    """Create train and validation dataloaders for copy task."""
    train_dataset = CopyTaskDataset(num_train, seq_len, vocab_size, blank_len, seed=42)
    val_dataset = CopyTaskDataset(num_val, seq_len, vocab_size, blank_len, seed=123)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, vocab_size


def evaluate_copy_accuracy(predictions, targets, blank_token=0):
    """
    Compute accuracy on the relevant (non-blank) parts.
    
    Only evaluates positions where target is not blank.
    """
    # predictions, targets: (batch, seq_len)
    mask = (targets != blank_token)
    correct = (predictions == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
    return accuracy
