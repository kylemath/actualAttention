"""
Shared utilities for experiments.
"""
import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

try:
    from config import DEVICE, PATHS, FIGURE_CONFIG
except ImportError:  # Allow package import
    from .config import DEVICE, PATHS, FIGURE_CONFIG


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get computation device."""
    return DEVICE


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def save_results(results: Dict, filename: str, base_dir: Optional[str] = None):
    """Save experimental results."""
    base = base_dir or PATHS['data']
    path = Path(base) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    
    # Also save as JSON for easy inspection
    json_path = path.with_suffix('.json')
    with open(json_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in results.items()
        }
        json.dump(json_results, f, indent=2)


def load_results(filename: str) -> Dict:
    """Load experimental results."""
    path = Path(PATHS['data']) / filename
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def compute_attention_sparsity(attention_weights: torch.Tensor, threshold: float = 0.01) -> float:
    """
    Compute sparsity of attention weights.
    
    Args:
        attention_weights: Tensor of shape (batch, heads, seq_len, seq_len)
        threshold: Values below this are considered zero
    
    Returns:
        Sparsity as fraction of near-zero values
    """
    return (attention_weights < threshold).float().mean().item()


def compute_attention_entropy(attention_weights: torch.Tensor) -> float:
    """
    Compute entropy of attention distributions.
    
    Higher entropy = more uniform attention (less focused)
    Lower entropy = more peaked attention (more focused)
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    attention_weights = attention_weights + eps
    entropy = -(attention_weights * torch.log(attention_weights)).sum(dim=-1)
    return entropy.mean().item()


def compute_variance_explained(attention_matrix: np.ndarray, k: int = 5) -> Tuple[np.ndarray, float]:
    """
    Compute variance explained by top-k eigenvalues.
    
    Returns:
        eigenvalues: Top-k eigenvalues
        variance_explained: Cumulative variance explained by top-k
    """
    # Compute eigenvalues via SVD (more stable)
    U, S, Vh = np.linalg.svd(attention_matrix)
    eigenvalues = S ** 2
    total_variance = eigenvalues.sum()
    top_k_variance = eigenvalues[:k].sum()
    variance_explained = top_k_variance / total_variance
    return eigenvalues[:k], variance_explained


def setup_plotting():
    """Setup matplotlib style for publication-quality figures."""
    plt.style.use(FIGURE_CONFIG['style'])
    sns.set_palette(FIGURE_CONFIG['color_palette'])
    plt.rcParams['font.size'] = FIGURE_CONFIG['font_size']
    plt.rcParams['figure.dpi'] = FIGURE_CONFIG['dpi']
    plt.rcParams['savefig.dpi'] = FIGURE_CONFIG['dpi']
    plt.rcParams['savefig.format'] = FIGURE_CONFIG['format']


def save_figure(fig, filename: str):
    """Save figure to paper/figures/ directory."""
    path = Path(PATHS['figures']) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=FIGURE_CONFIG['dpi'])
    plt.close(fig)


class MetricsLogger:
    """Track and log training metrics."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
    
    def log(self, step: int, **kwargs):
        """Log metrics at a given step."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = {'steps': [], 'values': []}
            self.metrics[key]['steps'].append(step)
            self.metrics[key]['values'].append(value)
    
    def save(self, filename: str = 'metrics.json'):
        """Save all logged metrics."""
        path = self.log_dir / filename
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, filename: str = 'metrics.json'):
        """Load logged metrics."""
        path = self.log_dir / filename
        with open(path, 'r') as f:
            self.metrics = json.load(f)


class ProgressTracker:
    """Track experimental progress across multiple runs."""
    
    def __init__(self, total_experiments: int):
        self.total = total_experiments
        self.completed = 0
        self.results = []
    
    def update(self, result: Dict):
        """Update progress with a completed experiment."""
        self.completed += 1
        self.results.append(result)
        print(f"Progress: {self.completed}/{self.total} experiments completed "
              f"({100*self.completed/self.total:.1f}%)")
    
    def summary(self):
        """Print summary statistics."""
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        for result in self.results:
            print(f"{result['name']}: Loss = {result.get('final_loss', 'N/A'):.4f}")
        print("="*50 + "\n")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def estimate_compute_cost(num_params: int, num_tokens: int, num_steps: int) -> float:
    """
    Estimate compute cost in FLOPs.
    
    Rough approximation: 2 * params * tokens * steps
    (forward + backward pass)
    """
    return 2 * num_params * num_tokens * num_steps
