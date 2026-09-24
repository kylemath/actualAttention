"""Analysis utilities."""
from .variance_analysis import compute_variance_explained, summarize_variance
from .gradient_flow import compute_layerwise_gradient_norms, summarize_gradient_flow
from .sparsity_analysis import compute_sparsity, compute_entropy

__all__ = [
    "compute_variance_explained",
    "summarize_variance",
    "compute_layerwise_gradient_norms",
    "summarize_gradient_flow",
    "compute_sparsity",
    "compute_entropy",
]
