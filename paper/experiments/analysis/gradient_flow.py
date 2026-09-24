"""
Gradient flow analysis utilities.
"""
import numpy as np


def compute_layerwise_gradient_norms(gradients):
    """
    Compute layerwise gradient norms from a list of gradient tensors.

    Args:
        gradients: List of numpy arrays or tensors (per-layer gradients)

    Returns:
        List of L2 norms for each layer.
    """
    norms = []
    for grad in gradients:
        g = grad.detach().cpu().numpy() if hasattr(grad, "detach") else grad
        norms.append(float(np.linalg.norm(g)))
    return norms


def summarize_gradient_flow(gradient_norms):
    """
    Summarize gradient flow across layers.
    """
    gradient_norms = np.array(gradient_norms)
    return {
        "mean": float(gradient_norms.mean()),
        "std": float(gradient_norms.std()),
        "min": float(gradient_norms.min()),
        "max": float(gradient_norms.max()),
        "decay_ratio": float(gradient_norms[-1] / (gradient_norms[0] + 1e-8)),
    }
