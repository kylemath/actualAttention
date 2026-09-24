"""
LSTM models: Standard and Bio-enhanced.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class StandardLSTM(nn.Module):
    """Standard LSTM."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc_out(output)
        return output, hidden


class BioEnhancedLSTM(nn.Module):
    """
    Bio-enhanced LSTM with:
    - Lateral inhibition on gates
    - Tuning sharpening on gate activations
    - Biased competition on input gate
    - Decorrelation on cell state dimensions
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
            'tuning_sharpening': {'beta': 1.5},
            'biased_competition': {'beta': 0.25, 'learnable': True},
            'noise_decorrelation': {'lambda': 0.01},
        }
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Use LSTMCell for manual control over gate computations
        self.lstm_cells = nn.ModuleList([
            BioEnhancedLSTMCell(
                embedding_dim if i == 0 else hidden_dim,
                hidden_dim,
                self.bio_config
            )
            for i in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.decorrelation_loss = 0.0
    
    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        batch_size, seq_len = x.size()
        
        # Initialize hidden state if not provided
        if hidden is None:
            h = [torch.zeros(batch_size, self.hidden_dim, device=x.device)
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_dim, device=x.device)
                 for _ in range(self.num_layers)]
            hidden = (h, c)
        
        h_states, c_states = hidden
        embedded = self.dropout(self.embedding(x))
        
        # Process sequence step by step
        outputs = []
        self.decorrelation_loss = 0.0
        
        for t in range(seq_len):
            input_t = embedded[:, t, :]
            
            # Process through LSTM layers
            for layer_idx, lstm_cell in enumerate(self.lstm_cells):
                h_new, c_new = lstm_cell(input_t, (h_states[layer_idx], c_states[layer_idx]))
                h_states[layer_idx] = h_new
                c_states[layer_idx] = c_new
                input_t = self.dropout(h_new)
                
                # Accumulate decorrelation loss
                self.decorrelation_loss += lstm_cell.decorrelation_loss
            
            outputs.append(h_states[-1])
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        output = self.fc_out(output)
        
        return output, (h_states, c_states)
    
    def get_total_loss(self, ce_loss):
        """Combine cross-entropy loss with decorrelation loss."""
        lambda_decorr = self.bio_config['noise_decorrelation']['lambda']
        return ce_loss + lambda_decorr * self.decorrelation_loss


class BioEnhancedLSTMCell(nn.Module):
    """Single LSTM cell with biological mechanisms."""
    
    def __init__(self, input_size: int, hidden_size: int, bio_config: dict):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bio_config = bio_config
        
        # Standard LSTM gates
        self.W_i = nn.Linear(input_size, hidden_size)  # Input gate
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.W_f = nn.Linear(input_size, hidden_size)  # Forget gate
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.W_o = nn.Linear(input_size, hidden_size)  # Output gate
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.W_c = nn.Linear(input_size, hidden_size)  # Cell candidate
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Learnable task bias for input gate (biased competition)
        if bio_config['biased_competition']['learnable']:
            self.task_bias = nn.Parameter(torch.randn(hidden_size) * 0.01)
        else:
            self.register_buffer('task_bias', torch.zeros(hidden_size))
        
        self.decorrelation_loss = 0.0
    
    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        
        # Compute gate pre-activations
        z_i = self.W_i(x) + self.U_i(h_prev)
        z_f = self.W_f(x) + self.U_f(h_prev)
        z_o = self.W_o(x) + self.U_o(h_prev)
        z_c = self.W_c(x) + self.U_c(h_prev)
        
        # === LATERAL INHIBITION: Subtract mean from each gate ===
        alpha = self.bio_config['lateral_inhibition']['alpha']
        z_i = z_i - alpha * z_i.mean(dim=-1, keepdim=True)
        z_f = z_f - alpha * z_f.mean(dim=-1, keepdim=True)
        z_o = z_o - alpha * z_o.mean(dim=-1, keepdim=True)
        
        # === TUNING SHARPENING: Scale gate activations ===
        beta = self.bio_config['tuning_sharpening']['beta']
        z_i = beta * z_i
        z_f = beta * z_f
        
        # === BIASED COMPETITION: Add task bias to input gate ===
        beta_bias = self.bio_config['biased_competition']['beta']
        z_i = z_i + beta_bias * self.task_bias
        
        # Apply gate activations
        i_gate = torch.sigmoid(z_i)
        f_gate = torch.sigmoid(z_f)
        o_gate = torch.sigmoid(z_o)
        c_candidate = torch.tanh(z_c)
        
        # Update cell state
        c_new = f_gate * c_prev + i_gate * c_candidate
        
        # === NOISE DECORRELATION: Penalize correlated dimensions ===
        self.decorrelation_loss = self._compute_decorrelation_loss(c_new)
        
        # Update hidden state
        h_new = o_gate * torch.tanh(c_new)
        
        return h_new, c_new
    
    def _compute_decorrelation_loss(self, c):
        """
        Compute decorrelation loss for cell state dimensions.
        
        Encourages different dimensions to encode independent information.
        """
        # c: (batch, hidden_dim)
        
        # Normalize
        c_norm = F.normalize(c, p=2, dim=0)  # Normalize across batch
        
        # Compute correlation matrix
        corr_matrix = torch.mm(c_norm.t(), c_norm) / c.size(0)
        
        # Remove diagonal (self-correlation)
        mask = torch.eye(self.hidden_size, device=c.device).bool()
        corr_matrix = corr_matrix.masked_fill(mask, 0.0)
        
        # Mean absolute correlation (we want this to be low)
        decorr_loss = corr_matrix.abs().mean()
        
        return decorr_loss


class StandardLSTMClassifier(nn.Module):
    """Standard LSTM classifier for continuous sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        output, hidden = self.lstm(x, hidden)
        pooled = output.mean(dim=1)
        logits = self.fc_out(self.dropout(pooled))
        return logits, hidden


class BioEnhancedLSTMClassifier(nn.Module):
    """Bio-enhanced LSTM classifier for continuous sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int,
                 dropout: float = 0.1, bio_config: Optional[dict] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bio_config = bio_config or {
            'lateral_inhibition': {'alpha': 0.3},
            'tuning_sharpening': {'beta': 1.5},
            'biased_competition': {'beta': 0.25, 'learnable': True},
            'noise_decorrelation': {'lambda': 0.01},
        }

        self.lstm_cells = nn.ModuleList([
            BioEnhancedLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, self.bio_config)
            for i in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.decorrelation_loss = 0.0

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            h = [torch.zeros(batch_size, self.hidden_dim, device=x.device)
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_dim, device=x.device)
                 for _ in range(self.num_layers)]
            hidden = (h, c)

        h_states, c_states = hidden
        outputs = []
        self.decorrelation_loss = 0.0

        for t in range(seq_len):
            input_t = x[:, t, :]
            for layer_idx, lstm_cell in enumerate(self.lstm_cells):
                h_new, c_new = lstm_cell(input_t, (h_states[layer_idx], c_states[layer_idx]))
                h_states[layer_idx] = h_new
                c_states[layer_idx] = c_new
                input_t = self.dropout(h_new)
                self.decorrelation_loss += lstm_cell.decorrelation_loss
            outputs.append(h_states[-1])

        output = torch.stack(outputs, dim=1)
        pooled = output.mean(dim=1)
        logits = self.fc_out(self.dropout(pooled))
        return logits, (h_states, c_states)

    def get_total_loss(self, ce_loss):
        lambda_decorr = self.bio_config['noise_decorrelation']['lambda']
        return ce_loss + lambda_decorr * self.decorrelation_loss
