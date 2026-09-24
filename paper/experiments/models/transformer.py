"""
Transformer models: Standard and Bio-enhanced.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class StandardTransformer(nn.Module):
    """Standard multi-head self-attention transformer."""
    
    def __init__(self, vocab_size: int, d_model: int, nhead: int, 
                 num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.attention_weights = []  # Store for analysis
        
    def forward(self, src, src_mask=None):
        # src: (batch, seq_len)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer(src, src_mask)
        output = self.fc_out(output)
        return output


class BioEnhancedTransformer(nn.Module):
    """
    Bio-enhanced transformer with:
    - Biased competition (task-relevant bias)
    - Lateral inhibition (row-wise centering)
    - Tuning sharpening (temperature scaling)
    - Noise decorrelation (multi-head diversity)
    """
    
    def __init__(self, vocab_size: int, d_model: int, nhead: int, 
                 num_layers: int, dim_feedforward: int, dropout: float = 0.1,
                 bio_config: Optional[dict] = None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Default bio-mechanism parameters
        self.bio_config = bio_config or {
            'biased_competition': {'beta': 0.25, 'learnable': True},
            'lateral_inhibition': {'alpha': 0.3},
            'tuning_sharpening': {'beta': 1.5, 'adaptive': True},
            'noise_decorrelation': {'lambda': 0.05},
        }
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Bio-enhanced transformer layers
        self.layers = nn.ModuleList([
            BioEnhancedTransformerLayer(
                d_model, nhead, dim_feedforward, dropout, self.bio_config
            )
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Learnable task bias (if enabled)
        if self.bio_config['biased_competition']['learnable']:
            self.task_bias = nn.Parameter(torch.randn(1, 1, d_model) * 0.01)
        else:
            self.register_buffer('task_bias', torch.zeros(1, 1, d_model))
        
        self.attention_weights = []  # Store for analysis
        self.diversity_loss = 0.0
        
    def forward(self, src, src_mask=None):
        # src: (batch, seq_len)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Add task bias (biased competition)
        src = src + self.task_bias
        
        # Reset diversity loss
        self.diversity_loss = 0.0
        
        # Pass through bio-enhanced layers
        for layer in self.layers:
            src = layer(src, src_mask)
            self.diversity_loss += layer.diversity_loss
        
        output = self.fc_out(src)
        return output
    
    def get_total_loss(self, ce_loss):
        """Combine cross-entropy loss with diversity loss."""
        noise_cfg = self.bio_config.get('noise_decorrelation', {})
        lambda_diversity = noise_cfg.get('lambda_', noise_cfg.get('lambda', 0.0))
        return ce_loss + lambda_diversity * self.diversity_loss


class BioEnhancedTransformerLayer(nn.Module):
    """Single layer of bio-enhanced transformer."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, 
                 dropout: float, bio_config: dict):
        super().__init__()
        self.self_attn = BioEnhancedMultiHeadAttention(
            d_model, nhead, dropout, bio_config
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.diversity_loss = 0.0
    
    def forward(self, src, src_mask=None):
        # Self-attention with residual
        src2, diversity_loss = self.self_attn(src, src, src, src_mask)
        self.diversity_loss = diversity_loss
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward with residual
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class BioEnhancedMultiHeadAttention(nn.Module):
    """Multi-head attention with biological mechanisms."""
    
    def __init__(self, d_model: int, nhead: int, dropout: float, bio_config: dict):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.bio_config = bio_config
        
        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable sharpening factor (if adaptive)
        if bio_config['tuning_sharpening']['adaptive']:
            self.sharpening_beta = nn.Parameter(
                torch.tensor(bio_config['tuning_sharpening']['beta'])
            )
        else:
            self.register_buffer(
                'sharpening_beta',
                torch.tensor(bio_config['tuning_sharpening']['beta'])
            )
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape to multi-head
        Q = self.q_linear(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # Compute attention scores: Q·K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # === LATERAL INHIBITION: Subtract row mean ===
        alpha = self.bio_config['lateral_inhibition']['alpha']
        row_means = scores.mean(dim=-1, keepdim=True)
        scores = scores - alpha * row_means
        
        # === TUNING SHARPENING: Scale scores ===
        scores = scores * self.sharpening_beta
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # === NOISE DECORRELATION: Compute diversity loss ===
        diversity_loss = self._compute_diversity_loss(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(output)
        
        return output, diversity_loss
    
    def _compute_diversity_loss(self, attn_weights):
        """
        Compute diversity loss to encourage different heads to attend to different patterns.
        
        Loss = mean pairwise correlation between attention patterns of different heads.
        """
        # attn_weights: (batch, nhead, seq_len, seq_len)
        batch_size, nhead, seq_len, _ = attn_weights.shape
        
        # Flatten attention patterns
        attn_flat = attn_weights.view(batch_size, nhead, -1)  # (batch, nhead, seq_len^2)
        
        # Normalize
        attn_norm = F.normalize(attn_flat, p=2, dim=-1)
        
        # Compute pairwise correlations
        # correlations: (batch, nhead, nhead)
        correlations = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
        
        # Remove diagonal (self-correlation)
        mask = torch.eye(nhead, device=attn_weights.device).bool()
        correlations = correlations.masked_fill(mask.unsqueeze(0), 0.0)
        
        # Mean absolute correlation (we want this to be low)
        diversity_loss = correlations.abs().mean()
        
        return diversity_loss


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal mask for decoder (lower triangular)."""
    # True indicates positions that should be masked
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()


class StandardTransformerClassifier(nn.Module):
    """Transformer encoder for sequence classification."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, src_mask=None):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_mask)
        # Pool over sequence (mean pooling)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


class BioEnhancedTransformerClassifier(nn.Module):
    """Bio-enhanced transformer encoder for sequence classification."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int,
        dropout: float = 0.1,
        bio_config: Optional[dict] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.bio_config = bio_config or {
            "biased_competition": {"beta": 0.25, "learnable": True},
            "lateral_inhibition": {"alpha": 0.3},
            "tuning_sharpening": {"beta": 1.5, "adaptive": True},
            "noise_decorrelation": {"lambda": 0.05},
        }

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.layers = nn.ModuleList(
            [
                BioEnhancedTransformerLayer(
                    d_model, nhead, dim_feedforward, dropout, self.bio_config
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Linear(d_model, num_classes)

        if self.bio_config["biased_competition"]["learnable"]:
            self.task_bias = nn.Parameter(torch.randn(1, 1, d_model) * 0.01)
        else:
            self.register_buffer("task_bias", torch.zeros(1, 1, d_model))

        self.diversity_loss = 0.0

    def forward(self, x, src_mask=None):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = x + self.task_bias

        self.diversity_loss = 0.0
        for layer in self.layers:
            x = layer(x, src_mask)
            self.diversity_loss += layer.diversity_loss

        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

    def get_total_loss(self, ce_loss):
        noise_cfg = self.bio_config.get("noise_decorrelation", {})
        lambda_diversity = noise_cfg.get("lambda_", noise_cfg.get("lambda", 0.0))
        return ce_loss + lambda_diversity * self.diversity_loss
