"""
Lateral inhibition mechanism.
"""
import torch


def apply_lateral_inhibition(tensor: torch.Tensor, alpha: float):
    """
    Subtract mean activation (centering).
    """
    mean = tensor.mean(dim=-1, keepdim=True)
    return tensor - alpha * mean
