"""
Variance analysis for attention matrices and hidden states.
"""
import numpy as np


def compute_variance_explained(matrix: np.ndarray, k: int = 5):
    """
    Compute variance explained by top-k components using SVD.
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    eigenvalues = S ** 2
    total = eigenvalues.sum() + 1e-8
    top_k = eigenvalues[:k].sum()
    return eigenvalues[:k], top_k / total


def summarize_variance(attention_matrices, k: int = 5):
    """
    Summarize variance explained across multiple attention matrices.
    """
    variances = []
    for mat in attention_matrices:
        _, v = compute_variance_explained(mat, k=k)
        variances.append(v)
    return float(np.mean(variances)), float(np.std(variances))
