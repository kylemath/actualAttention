"""
Tuning sharpening mechanism.
"""
import torch


def apply_tuning_sharpening(tensor: torch.Tensor, beta: float):
    """
    Scale activations to sharpen tuning.
    """
    return tensor * beta
