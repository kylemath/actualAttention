"""
Noise decorrelation mechanism.
"""
import torch
import torch.nn.functional as F


def decorrelation_loss(tensor: torch.Tensor):
    """
    Compute a simple decorrelation loss for a 2D tensor (batch, features).
    """
    # Normalize across batch
    normed = F.normalize(tensor, p=2, dim=0)
    corr = torch.mm(normed.t(), normed) / tensor.size(0)
    mask = torch.eye(corr.size(0), device=corr.device).bool()
    corr = corr.masked_fill(mask, 0.0)
    return corr.abs().mean()
