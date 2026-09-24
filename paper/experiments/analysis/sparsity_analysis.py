"""
Sparsity analysis for attention matrices.
"""
import numpy as np


def compute_sparsity(attention_matrix: np.ndarray, threshold: float = 0.01) -> float:
    """
    Compute sparsity as fraction of values below threshold.
    """
    return float((np.abs(attention_matrix) < threshold).mean())


def compute_entropy(attention_matrix: np.ndarray) -> float:
    """
    Compute entropy of attention distribution (row-wise).
    """
    eps = 1e-8
    attn = attention_matrix + eps
    attn = attn / attn.sum(axis=-1, keepdims=True)
    entropy = -(attn * np.log(attn)).sum(axis=-1)
    return float(entropy.mean())
