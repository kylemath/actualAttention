"""
RNN models: Standard and Bio-enhanced.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StandardRNN(nn.Module):
    """Standard vanilla RNN."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            embedding_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc_out(output)
        return output, hidden


class BioEnhancedRNN(nn.Module):
    """
    Bio-enhanced RNN with:
    - Lateral inhibition on hidden states
    - Tuning sharpening on hidden activations
    - Biased competition via task-relevant bias
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.1,
                 bio_config: Optional[dict] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Default bio-mechanism parameters
        self.bio_config = bio_config or {
            'lateral_inhibition': {'alpha': 0.3},
            'tuning_sharpening': {'beta': 1.2},
            'biased_competition': {'beta': 0.1, 'learnable': True},
        }
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Use RNNCell for manual control over hidden state updates
        self.rnn_cells = nn.ModuleList([
            nn.RNNCell(embedding_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable task bias (if enabled)
        if self.bio_config['biased_competition']['learnable']:
            self.task_bias = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        else:
            self.register_buffer('task_bias', torch.zeros(hidden_dim))
    
    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        batch_size, seq_len = x.size()
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = [torch.zeros(batch_size, self.hidden_dim, device=x.device)
                     for _ in range(self.num_layers)]
        
        embedded = self.dropout(self.embedding(x))
        
        # Process sequence step by step
        outputs = []
        for t in range(seq_len):
            input_t = embedded[:, t, :]
            
            # Process through RNN layers with bio-mechanisms
            for layer_idx, rnn_cell in enumerate(self.rnn_cells):
                h = rnn_cell(input_t, hidden[layer_idx])
                
                # === LATERAL INHIBITION: Subtract mean activation ===
                alpha = self.bio_config['lateral_inhibition']['alpha']
                h_mean = h.mean(dim=-1, keepdim=True)
                h = h - alpha * h_mean
                
                # === BIASED COMPETITION: Add task bias ===
                beta_bias = self.bio_config['biased_competition']['beta']
                h = h + beta_bias * self.task_bias
                
                # === TUNING SHARPENING: Scale activations ===
                beta_sharp = self.bio_config['tuning_sharpening']['beta']
                h = beta_sharp * h
                
                # Apply tanh activation
                h = torch.tanh(h)
                
                hidden[layer_idx] = h
                input_t = self.dropout(h)
            
            outputs.append(hidden[-1])
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        output = self.fc_out(output)
        
        return output, hidden


class StandardGRU(nn.Module):
    """Standard GRU (for comparison)."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded, hidden)
        output = self.fc_out(output)
        return output, hidden


class StandardRNNClassifier(nn.Module):
    """Standard RNN classifier for continuous sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_dim)
        output, hidden = self.rnn(x, hidden)
        pooled = output.mean(dim=1)
        logits = self.fc_out(self.dropout(pooled))
        return logits, hidden


class BioEnhancedRNNClassifier(nn.Module):
    """Bio-enhanced RNN classifier for continuous sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int,
                 dropout: float = 0.1, bio_config: Optional[dict] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bio_config = bio_config or {
            'lateral_inhibition': {'alpha': 0.3},
            'tuning_sharpening': {'beta': 1.2},
            'biased_competition': {'beta': 0.1, 'learnable': True},
        }

        self.rnn_cells = nn.ModuleList([
            nn.RNNCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        if self.bio_config['biased_competition']['learnable']:
            self.task_bias = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        else:
            self.register_buffer('task_bias', torch.zeros(hidden_dim))

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            hidden = [torch.zeros(batch_size, self.hidden_dim, device=x.device)
                      for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :]
            for layer_idx, rnn_cell in enumerate(self.rnn_cells):
                h = rnn_cell(input_t, hidden[layer_idx])

                alpha = self.bio_config['lateral_inhibition']['alpha']
                h = h - alpha * h.mean(dim=-1, keepdim=True)

                beta_bias = self.bio_config['biased_competition']['beta']
                h = h + beta_bias * self.task_bias

                beta_sharp = self.bio_config['tuning_sharpening']['beta']
                h = beta_sharp * h

                h = torch.tanh(h)
                hidden[layer_idx] = h
                input_t = self.dropout(h)

            outputs.append(hidden[-1])

        output = torch.stack(outputs, dim=1)
        pooled = output.mean(dim=1)
        logits = self.fc_out(self.dropout(pooled))
        return logits, hidden
