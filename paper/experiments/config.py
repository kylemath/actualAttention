"""
Shared configuration for all experiments.
"""
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

# Experiment modes
MODES = {
    'parallel': {
        'epochs': 30,
        'batch_size': 32,
        'num_sequences': 2000,
        'sequence_length': 128,
        'model_dim': 256,
        'num_heads': 4,
    },
    'fast': {
        'epochs': 20,
        'batch_size': 32,
        'num_sequences': 1000,
        'sequence_length': 128,
        'model_dim': 256,
        'num_heads': 4,
    },
    'cycle': {
        'epochs': 2,
        'batch_size': 16,
        'num_sequences': 200,
        'sequence_length': 64,
        'model_dim': 128,
        'num_heads': 2,
    },
    'full': {
        'epochs': 100,
        'batch_size': 64,
        'num_sequences': 10000,
        'sequence_length': 256,
        'model_dim': 512,
        'num_heads': 8,
    },
    'large': {
        'epochs': 200,
        'batch_size': 128,
        'num_sequences': 50000,
        'sequence_length': 512,
        'model_dim': 1024,
        'num_heads': 16,
    }
}

# Model hyperparameters
RNN_CONFIG = {
    'hidden_dim': 512,
    'num_layers': 2,
    'dropout': 0.1,
}

LSTM_CONFIG = {
    'hidden_dim': 512,
    'num_layers': 2,
    'dropout': 0.1,
}

TRANSFORMER_CONFIG = {
    'num_layers': 6,
    'num_heads': 8,
    'dim_feedforward': 2048,
    'dropout': 0.1,
}

# Biological mechanism parameters
BIO_MECHANISMS = {
    'biased_competition': {
        'beta': 0.25,  # bias strength
        'learnable': True,  # whether bias is learned or fixed
    },
    'lateral_inhibition': {
        'alpha': 0.3,  # inhibition strength
        'radius': None,  # None = global, int = local neighborhood
    },
    'tuning_sharpening': {
        'beta': 1.5,  # sharpening factor (>1 = sharper)
        'adaptive': True,  # adapt based on entropy
    },
    'noise_decorrelation': {
        'lambda_': 0.05,  # decorrelation loss weight
        'method': 'orthogonal',  # 'orthogonal' or 'diverse'
    },
}

# Training hyperparameters
TRAINING = {
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,
    'warmup_steps': 1000,
    'lr_scheduler': 'cosine',
    'optimizer': 'adamw',
}

# Task configurations
TASKS = {
    'copy': {
        'vocab_size': 10,
        'sequence_length': 128,
        'blank_length': 10,
    },
    'language_model': {
        'vocab_size': 256,  # character-level
        'sequence_length': 256,
        'dataset': 'enwik8',
    },
    'sequential_mnist': {
        'input_dim': 784,
        'num_classes': 10,
        'permute': False,
    },
}

# Evaluation metrics
METRICS = [
    'loss',
    'accuracy',
    'perplexity',
    'gradient_norm',
    'attention_entropy',
    'attention_sparsity',
    'variance_explained',
    'memory_usage',
    'throughput',
]

# Analysis parameters
ANALYSIS = {
    'save_checkpoints': True,
    'checkpoint_freq': 10,  # epochs
    'log_freq': 100,  # steps
    'visualize_attention': True,
    'compute_eigenvalues': True,
    'track_gradients': True,
}

# Output paths (relative to paper/ directory)
PATHS = {
    'data': 'data',
    'figures': 'figures',
    'checkpoints': 'checkpoints',
    'logs': 'logs',
}

# Random seeds for reproducibility
SEEDS = [42, 123, 456, 789, 1337]  # Run 5 seeds for statistical significance

# Figure style
FIGURE_CONFIG = {
    'dpi': 300,
    'format': 'pdf',
    'figsize': (8, 6),
    'font_size': 12,
    'style': 'seaborn-v0_8-paper',
    'color_palette': 'Set2',
}

# Statistical testing
STATS = {
    'confidence_level': 0.95,
    'test': 'paired_ttest',  # paired t-test for same architectures
    'correction': 'bonferroni',  # multiple comparison correction
}
