"""
Training script for Mini GPT
Main entry point for training the model
"""

import torch
import os
import time
from config import GPTConfig
from tokenizer import CharTokenizer
from model import MiniGPT
from utils import (
    download_dataset, 
    prepare_data, 
    get_batch, 
    estimate_loss, 
    get_lr,
    calculate_perplexity
)


def train():
    """
    Main training loop for Mini GPT.
    
    Steps:
    1. Setup: Load config, dataset, tokenizer
    2. Initialize model and optimizer
    3. Training loop with evaluation
    4. Save checkpoints
    """
    
    # ============ Setup ============
    print("=" * 60)
    print("Mini GPT Training")
    print("=" * 60)
    
    # Load configuration
    config = GPTConfig()
    
    # Set device (CUDA, MPS for Mac M1/M2, or CPU)
    if torch.cuda.is_available():
        config.device = 'cuda'
    elif torch.backends.mps.is_available():
        config.device = 'mps'
    else:
        config.device = 'cpu'
    print(f"Using device: {config.device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    
    # Download and load dataset
    print("\n" + "=" * 60)
    print("Loading Dataset")
    print("=" * 60)
    text = download_dataset(config.dataset_name)
    
    # Initialize tokenizer and build vocabulary
    print("\n" + "=" * 60)
    print("Building Tokenizer")
    print("=" * 60)
    tokenizer = CharTokenizer(text)
    
    # Update config with actual vocab size
    config.vocab_size = tokenizer.vocab_size
    
    # Prepare train/val splits
    train_data, val_data = prepare_data(text, tokenizer, config.train_split)
    
    # ============ Model Initialization ============
    print("\n" + "=" * 60)
    print("Initializing Model")
    print("=" * 60)
    print(config)
    print()
    
    model = MiniGPT(config)
    model = model.to(config.device)
    
    # ============ Optimizer Setup ============
    # AdamW optimizer with weight decay for regularization
    # Why AdamW?
    # - Adaptive learning rates per parameter
    # - Better weight decay than Adam
    # - Standard for training transformers
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # ============ Training Loop ============
    print("\n" + "=" * 60)
    print("Training Started")
    print("=" * 60)
    
    # Create checkpoint directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Training state
    best_val_loss = float('inf')
    start_time = time.time()
    
    for iteration in range(config.max_iters):
        
        # -------- Evaluation --------
        if iteration % config.eval_interval == 0 or iteration == config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            train_loss = losses['train']
            val_loss = losses['val']
            
            # Calculate perplexity
            train_ppl = calculate_perplexity(train_loss)
            val_ppl = calculate_perplexity(val_loss)
            
            # Time elapsed
            elapsed = time.time() - start_time
            
            print(f"Iter {iteration:5d} | "
                  f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:6.2f}) | "
                  f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:6.2f}) | "
                  f"Time: {elapsed:.1f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config,
                }
                torch.save(checkpoint, os.path.join(config.save_dir, 'best_model.pt'))
                print(f"  → Saved best model (val_loss: {val_loss:.4f})")
            
            # Generate sample text to monitor progress
            if iteration % (config.eval_interval * 2) == 0:
                print("\n  Sample generation:")
                generate_sample(model, tokenizer, config, prompt="ROMEO:")
                print()
        
        # -------- Training Step --------
        
        # Update learning rate (with warmup and cosine decay)
        if config.use_lr_scheduler:
            lr = get_lr(iteration, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Get batch
        x, y = get_batch(train_data, config.block_size, config.batch_size, config.device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        # Why clip?
        # - Prevents instability from occasional large gradients
        # - Especially important for RNNs/Transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        
        # -------- Periodic Checkpoint --------
        if iteration > 0 and iteration % config.save_interval == 0:
            checkpoint_path = os.path.join(config.save_dir, f'checkpoint_{iteration}.pt')
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, checkpoint_path)
            print(f"  → Saved checkpoint: {checkpoint_path}")
    
    # ============ Training Complete ============
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {calculate_perplexity(best_val_loss):.2f}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    # Final sample generation
    print("\n" + "=" * 60)
    print("Final Sample Generations")
    print("=" * 60)
    generate_sample(model, tokenizer, config, prompt="ROMEO:")
    print()
    generate_sample(model, tokenizer, config, prompt="First Citizen:")
    print()


@torch.no_grad()
def generate_sample(model, tokenizer, config, prompt="", max_tokens=200):
    """
    Generate and print a sample of text.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer instance
        config: Configuration
        prompt: Starting text
        max_tokens: Maximum tokens to generate
    """
    model.eval()
    
    # Encode prompt
    if prompt:
        tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=config.device)
        tokens = tokens.unsqueeze(0)  # Add batch dimension
    else:
        # Start with a random token
        tokens = torch.randint(0, config.vocab_size, (1, 1), device=config.device)
    
    # Generate
    generated = model.generate(
        tokens, 
        max_new_tokens=max_tokens,
        temperature=0.8,  # Some randomness
        top_k=40  # Sample from top 40 tokens
    )
    
    # Decode and print
    text = tokenizer.decode(generated[0].tolist())
    print(f"  {text}")
    
    model.train()


def load_and_generate(checkpoint_path, prompt="", max_tokens=500):
    """
    Load a trained model and generate text.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        prompt: Starting text
        max_tokens: Number of tokens to generate
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    
    config = checkpoint['config']
    
    # Set device
    if torch.cuda.is_available():
        config.device = 'cuda'
    elif torch.backends.mps.is_available():
        config.device = 'mps'
    else:
        config.device = 'cpu'
    
    # Load tokenizer (need to rebuild vocab from dataset)
    text = download_dataset(config.dataset_name)
    tokenizer = CharTokenizer(text)
    
    # Load model
    model = MiniGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()
    
    print(f"Model loaded (iteration {checkpoint['iteration']})")
    print("=" * 60)
    print("Generated text:")
    print("=" * 60)
    
    generate_sample(model, tokenizer, config, prompt=prompt, max_tokens=max_tokens)


if __name__ == '__main__':
    import sys
    
    # Check if we're loading a checkpoint for generation
    if len(sys.argv) > 1:
        if sys.argv[1] == 'generate':
            # Usage: python train.py generate checkpoints/best_model.pt "ROMEO:"
            checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else 'checkpoints/best_model.pt'
            prompt = sys.argv[3] if len(sys.argv) > 3 else ""
            load_and_generate(checkpoint_path, prompt)
        else:
            print("Usage:")
            print("  Train: python train.py")
            print("  Generate: python train.py generate <checkpoint_path> <optional_prompt>")
    else:
        # Normal training
        train()
