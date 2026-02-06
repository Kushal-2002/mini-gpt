# Mini GPT

A clean, from-scratch implementation of a decoder-only Transformer (GPT architecture) in PyTorch.

**Author:** Kushal Mukherjee

---

## 🎯 Overview

This project implements a GPT-style language model with:
- **Multi-head self-attention** with causal masking
- **Transformer decoder blocks** (6 layers, 6 heads)
- **Character-level tokenization**
- **~10M parameters**
- Training on TinyShakespeare dataset

**No HuggingFace. No pre-built modules. Pure PyTorch.**

---

## 🚀 Quick Start

### Install Dependencies
```bash
pip install torch requests
```

### Train the Model
```bash
python train.py
```

Training takes **~15-20 minutes** on Apple M4 or GPU. The script will:
- Download TinyShakespeare dataset automatically
- Train for 5000 iterations
- Save checkpoints to `checkpoints/`
- Display progress and sample generations

### Generate Text
```bash
python train.py generate checkpoints/best_model.pt "ROMEO:"
```

---

## 📁 Project Structure

```
├── config.py       # Hyperparameters (d_model=384, n_layers=6, etc.)
├── tokenizer.py    # Character-level tokenizer
├── model.py        # Core Transformer architecture
├── utils.py        # Dataset handling, LR scheduling
└── train.py        # Training loop and generation
```

---

## 🏗️ Architecture

**Model Components:**
- Token + Positional Embeddings (learned)
- 6 Transformer Blocks:
  - Multi-Head Self-Attention (6 heads, causal masking)
  - Feed-Forward Network (4x expansion)
  - Layer Normalization (Pre-LN)
  - Residual Connections
- Language Modeling Head (with weight tying)

**Key Hyperparameters:**
- `d_model`: 384
- `n_layers`: 6
- `n_heads`: 6
- `block_size`: 256 (context window)
- `batch_size`: 64
- `learning_rate`: 3e-4 with warmup + cosine decay

---

## 🎓 Key Features

1. **Complete from-scratch implementation**
   - Multi-head attention with causal masking
   - Positional encodings
   - Layer normalization and residuals

2. **Modern training practices**
   - AdamW optimizer
   - Learning rate warmup + cosine decay
   - Gradient clipping
   - Dropout regularization

3. **Text generation**
   - Temperature-based sampling
   - Top-k filtering
   - Autoregressive generation

---

## 📊 Results

After training:
- **Validation Loss:** ~1.50
- **Perplexity:** ~4.5
- **Generation Quality:** Coherent Shakespeare-style dialogue

**Sample Output:**
```
ROMEO:
What says my love? O, my dear lord,
The which he will revenge on me, and yet
I'll speak to my ghostly father.
```

---

## 🔧 Customization

Modify hyperparameters in `config.py`:
```python
n_layers = 6        # Transformer blocks
n_heads = 6         # Attention heads
d_model = 384       # Hidden dimension
max_iters = 5000    # Training iterations
```

---

## 📚 Understanding the Code

**Start here:**
1. `config.py` - See all hyperparameters
2. `model.py` - Core architecture (especially `MultiHeadSelfAttention`)
3. `train.py` - Training loop

**Key concepts implemented:**
- Scaled dot-product attention
- Causal masking for autoregressive modeling
- Pre-layer normalization
- Weight tying between embeddings and output

---

## 🚀 Next Steps

- Implement BPE tokenizer for efficiency
- Scale to larger datasets (WikiText-103)
- Add Flash Attention for speed
- Experiment with different architectures

---

**Built by Kushal Mukherjee**
