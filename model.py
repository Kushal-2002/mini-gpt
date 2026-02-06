"""
Mini GPT: Decoder-only Transformer implementation from scratch
This file contains all the core components of a GPT-style model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Why attention?
    - Allows model to focus on different parts of the sequence
    - Each position can attend to all previous positions (causal)
    - Captures long-range dependencies
    
    Why multi-head?
    - Different heads can learn different relationships
    - One head might focus on syntax, another on semantics
    - Increases model capacity without much compute cost
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        
        # Query, Key, Value projections for all heads (computed in parallel)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        
        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask: prevent attending to future positions
        # This makes it autoregressive (decoder-only)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        B, T, C = x.shape  # Batch, Time (sequence length), Channels (d_model)
        
        # Compute Q, K, V for all heads in batch
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.d_model, dim=2)  # Each is (B, T, C)
        
        # Split into multiple heads: (B, T, C) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # Why scale? Prevents softmax saturation with large d_k
        # Scores: (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask: set future positions to -inf before softmax
        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, 
            float('-inf')
        )
        
        # Convert scores to probabilities
        attn_weights = F.softmax(scores, dim=-1)  # (B, n_heads, T, T)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = attn_weights @ v  # (B, n_heads, T, head_dim)
        
        # Concatenate heads: (B, n_heads, T, head_dim) -> (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Why FFN?
    - Adds non-linearity and depth to the model
    - Applied independently to each position
    - Typically 4x expansion then contraction (d_model -> d_ff -> d_model)
    
    Architecture: Linear -> GELU -> Linear -> Dropout
    """
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, T, d_model)
        Returns:
            Output tensor (B, T, d_model)
        """
        # Why GELU instead of ReLU?
        # - Smoother gradients
        # - Used in GPT-2, BERT
        # - Better empirical performance
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer decoder block.
    
    Architecture:
    1. LayerNorm -> Multi-Head Attention -> Residual
    2. LayerNorm -> Feed-Forward -> Residual
    
    Why LayerNorm before (Pre-LN)?
    - Stabilizes training in deep networks
    - Better gradient flow
    - Allows training without learning rate warmup (though we still use it)
    """
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, T, d_model)
        Returns:
            Output tensor (B, T, d_model)
        """
        # Why residual connections?
        # - Enable gradient flow through deep networks
        # - Allow model to learn identity function easily
        # - Prevent degradation problem
        
        # Attention sub-layer with residual
        x = x + self.attn(self.ln1(x))
        
        # FFN sub-layer with residual
        x = x + self.ffn(self.ln2(x))
        
        return x


class MiniGPT(nn.Module):
    """
    Mini GPT: Decoder-only Transformer for autoregressive language modeling.
    
    Architecture Overview:
    1. Token + Position embeddings
    2. Stack of Transformer blocks
    3. LayerNorm
    4. Language modeling head (Linear projection to vocab)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        # Token embedding: maps token IDs to vectors
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional embedding: learned position encodings
        # Why learned instead of sinusoidal?
        # - Simpler to implement
        # - Can adapt to data
        # - Works well for fixed context lengths
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Language modeling head
        # Projects hidden states to vocabulary logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying: share weights between token embeddings and lm_head
        # Why? Reduces parameters and often improves performance
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model created with {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        """
        Initialize weights following GPT-2 paper.
        
        Why careful initialization?
        - Prevents exploding/vanishing gradients
        - Helps training stability
        - Impacts final performance
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx, targets=None):
        """
        Forward pass.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices (B, T) for computing loss
        
        Returns:
            If targets is None: logits (B, T, vocab_size)
            If targets provided: (logits, loss)
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        # Get embeddings
        # Token embeddings: what each token means
        tok_emb = self.token_embedding(idx)  # (B, T, d_model)
        
        # Position embeddings: where each token is
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.position_embedding(pos)  # (T, d_model)
        
        # Combine: sum token and position embeddings
        x = self.dropout(tok_emb + pos_emb)  # (B, T, d_model)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)  # (B, T, d_model)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) and (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore padding if we add it later
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Context tokens (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
        
        Returns:
            Generated sequence (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus on last time step and apply temperature
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Optionally crop to top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)
        
        return idx
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
