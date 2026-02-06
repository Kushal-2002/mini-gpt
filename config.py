"""
Configuration file for Mini GPT
Contains all hyperparameters and settings in one place for easy experimentation.
"""

class GPTConfig:
    """
    Configuration class for Mini GPT model.
    These hyperparameters control the model architecture and training process.
    """
    
    # Model Architecture
    vocab_size: int = 256  # Will be set dynamically based on tokenizer
    n_layers: int = 6  # Number of transformer blocks (depth)
    n_heads: int = 6   # Number of attention heads (must divide d_model evenly)
    d_model: int = 384  # Embedding dimension (hidden size)
    d_ff: int = 1536   # Feed-forward dimension (typically 4x d_model)
    block_size: int = 256  # Maximum sequence length (context window)
    dropout: float = 0.2  # Dropout probability for regularization
    
    # Training Hyperparameters
    batch_size: int = 64  # Number of sequences per batch
    learning_rate: float = 3e-4  # Initial learning rate (Adam default)
    max_iters: int = 5000  # Total training iterations
    eval_interval: int = 500  # Evaluate every N iterations
    eval_iters: int = 200  # Number of iterations for evaluation
    
    # Learning Rate Scheduler (meaningful modification)
    # Using cosine annealing with warm restarts for better convergence
    use_lr_scheduler: bool = True
    warmup_iters: int = 500  # Linear warmup steps
    lr_decay_iters: int = 5000  # Iterations for cosine decay
    min_lr: float = 3e-5  # Minimum learning rate (10% of max)
    
    # Dataset
    dataset_name: str = 'tinyshakespeare'  # 'tinyshakespeare' or 'tinystories'
    train_split: float = 0.9  # 90% train, 10% validation
    
    # System
    device: str = 'cuda'  # 'cuda' or 'cpu' or 'mps' (for M1/M2 Macs)
    seed: int = 42  # Random seed for reproducibility
    
    # Saving
    save_dir: str = 'checkpoints'
    save_interval: int = 1000  # Save checkpoint every N iterations
    
    def __repr__(self):
        """Pretty print configuration"""
        attrs = vars(self)
        return '\n'.join(f'{k}: {v}' for k, v in attrs.items())
