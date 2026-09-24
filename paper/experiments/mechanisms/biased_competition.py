"""
Biased competition mechanism.
"""
import torch


def apply_biased_competition(tensor: torch.Tensor, bias: torch.Tensor, beta: float):
    """
    Add task-relevant bias to a tensor (e.g., attention scores or hidden state).
    """
    return tensor + beta * bias
