"""
Utility functions for dataset handling and training
"""

import torch
import requests
import os


def download_dataset(dataset_name='tinyshakespeare'):
    """
    Download a small public dataset for training.
    
    Args:
        dataset_name: 'tinyshakespeare' or 'tinystories'
    
    Returns:
        str: The downloaded text
    """
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name == 'tinyshakespeare':
        # ~1MB text file, complete works of Shakespeare
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        file_path = os.path.join(data_dir, 'tinyshakespeare.txt')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Download if not exists
    if not os.path.exists(file_path):
        print(f"Downloading {dataset_name}...")
        response = requests.get(url)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Downloaded to {file_path}")
    else:
        print(f"Found existing dataset at {file_path}")
    
    # Read and return
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset size: {len(text):,} characters")
    return text


def prepare_data(text, tokenizer, train_split=0.9):
    """
    Prepare train and validation datasets.
    
    Args:
        text: Full text corpus
        tokenizer: Tokenizer instance
        train_split: Fraction of data for training
    
    Returns:
        train_data, val_data: Encoded PyTorch tensors
    """
    # Encode entire dataset
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Split train/val
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    
    return train_data, val_data


def get_batch(data, block_size, batch_size, device):
    """
    Generate a random batch of training data.
    
    Why random batches?
    - Provides data diversity during training
    - Acts as regularization
    - Efficient use of limited data
    
    Args:
        data: Encoded token tensor
        block_size: Sequence length
        batch_size: Number of sequences
        device: Device to place tensors on
    
    Returns:
        x: Input sequences (batch_size, block_size)
        y: Target sequences (batch_size, block_size)
    """
    # Random starting indices for each sequence
    # Ensure we can take block_size + 1 tokens (input + target)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Stack sequences
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    """
    Estimate loss on train and validation sets.
    
    Why estimate?
    - Full dataset evaluation is expensive
    - Averaging over multiple batches gives stable estimate
    - Provides unbiased estimate of true loss
    
    Args:
        model: The GPT model
        train_data: Training data tensor
        val_data: Validation data tensor
        config: Model configuration
    
    Returns:
        dict: {'train': train_loss, 'val': val_loss}
    """
    out = {}
    model.eval()  # Set to evaluation mode (disables dropout)
    
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config.eval_iters)
        
        for k in range(config.eval_iters):
            x, y = get_batch(data, config.block_size, config.batch_size, config.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    
    model.train()  # Set back to training mode
    return out


def get_lr(iteration, config):
    """
    Get learning rate with warmup and cosine decay.
    
    Learning rate schedule (meaningful modification):
    1. Linear warmup: 0 -> max_lr over warmup_iters
    2. Cosine decay: max_lr -> min_lr over lr_decay_iters
    
    Why this schedule?
    - Warmup prevents instability in early training
    - Cosine decay smoothly reduces lr for fine convergence
    - Used successfully in GPT-3, BERT, etc.
    
    Args:
        iteration: Current training iteration
        config: Configuration with lr settings
    
    Returns:
        float: Learning rate for this iteration
    """
    # 1. Linear warmup
    if iteration < config.warmup_iters:
        return config.learning_rate * iteration / config.warmup_iters
    
    # 2. After lr_decay_iters, return min_lr
    if iteration > config.lr_decay_iters:
        return config.min_lr
    
    # 3. Cosine decay between warmup and max iterations
    decay_ratio = (iteration - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    
    # Cosine annealing from 1.0 to 0.0
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio * 3.14159)))
    
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def calculate_perplexity(loss):
    """
    Calculate perplexity from loss.
    
    Perplexity = exp(loss)
    
    Why perplexity?
    - More interpretable than cross-entropy loss
    - Roughly: "the model is as confused as if choosing uniformly from N options"
    - Lower is better
    
    Args:
        loss: Cross-entropy loss
    
    Returns:
        float: Perplexity
    """
    return torch.exp(torch.tensor(loss)).item()
