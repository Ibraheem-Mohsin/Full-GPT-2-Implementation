# Full GPT-2 Implementation From Scratch

An educational **decoder-only Transformer / GPT-style language model** implemented from scratch in **PyTorch**, inspired by Andrej Karpathy’s nanoGPT.

This project was built to understand how modern language models work internally rather than only using high-level libraries. It includes the full training and generation pipeline, along with code-level exploration of important concepts such as:

- token and positional embeddings
- multi-head causal self-attention
- feedforward MLP blocks
- residual connections
- layer normalization
- weight tying
- GPT-style weight initialization
- mini-batch training
- AdamW optimization
- cosine learning rate scheduling with warmup
- gradient clipping
- autoregressive text generation

---

## Project Goals

The purpose of this project is to **learn and demonstrate the internals of GPT-style models** by implementing the architecture manually.

Instead of treating a Transformer as a black box, this repository shows how the model is built from the ground up:

1. raw text is tokenized
2. tokens are converted into embeddings
3. embeddings pass through stacked Transformer blocks
4. the model predicts the next token using a vocabulary projection head
5. training is done with cross-entropy loss and AdamW

This makes the code useful for:

- learning Transformer architecture
- understanding GPT-2 style implementations
- debugging training behavior
- experimenting with attention internals
- visualizing embeddings, attention scores, and masked attention matrices

---

## Architecture Overview

The model follows the standard **decoder-only Transformer** design used in GPT models.

### High-Level Flow

```text
Input Token IDs
   ↓
Token Embeddings
   +
Position Embeddings
   ↓
Stack of Transformer Blocks
   ↓
Final LayerNorm
   ↓
Linear Vocabulary Projection (lm_head)
   ↓
Logits over Vocabulary
```

### Transformer Block

Each Transformer block contains:

```text
LayerNorm
→ Causal Self-Attention
→ Residual Add
→ LayerNorm
→ MLP
→ Residual Add
```

### Causal Self-Attention

The attention module:

- projects the input into **Q, K, V**
- splits them into multiple heads
- applies **scaled dot-product attention**
- uses a **causal mask** so future tokens cannot be seen
- concatenates head outputs
- projects back to the embedding dimension

Attention formula:

```text
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

### Feedforward MLP

The MLP inside each Transformer block expands and compresses the embedding dimension:

```text
Linear(n_embd → 4*n_embd)
→ GELU
→ Linear(4*n_embd → n_embd)
```

This gives the model additional nonlinear expressive power.

---

## Default GPT Configuration

The default implementation matches **GPT-2 (Small) style dimensions**:

| Parameter | Value |
|---|---:|
| `block_size` | 1024 |
| `vocab_size` | 50257 |
| `n_layer` | 12 |
| `n_head` | 12 |
| `n_embd` | 768 |

This means:

- 12 Transformer blocks
- 12 attention heads
- embedding size of 768
- maximum sequence length of 1024 tokens

---

## Features Implemented

### 1. Token + Positional Embeddings
The model learns:

- **token embeddings** for vocabulary items
- **position embeddings** for token positions in the sequence

These are added together before entering the Transformer stack.

### 2. Multi-Head Causal Self-Attention
Implements GPT-style masked self-attention so every token can only attend to itself and previous tokens.

### 3. Weight Tying
The token embedding matrix and final output projection share weights:

```python
self.transformer.wte.weight = self.lm_head.weight
```

This reduces parameters and follows the GPT-2 design.

### 4. GPT-Style Weight Initialization
Uses normal initialization with special residual scaling for stability in deep residual networks.

### 5. Lightweight Data Loader
A simple custom data loader tokenizes `input.txt` and creates next-token prediction batches.


### 6. Gradient Clipping
Prevents exploding gradients during training.

### 7. Text Generation
Supports autoregressive generation with:

- prefix prompt input
- top-k sampling
- repeated generation of multiple continuations

### 8. Educational Probing / Inspection
The code can be extended to inspect internal tensors such as:

- token embeddings
- positional embeddings
- Q, K, V
- raw attention scores
- masked attention matrix
- attention weights after softmax
- MLP outputs

---

## Training Objective

This model is trained using **next-token prediction**.

For a token sequence:

```text
[t0, t1, t2, t3]
```

The input and targets are shifted:

```text
x = [t0, t1, t2]
y = [t1, t2, t3]
```

The loss is standard cross-entropy over the vocabulary.

---

## Optimization Details

Training uses **AdamW** with parameter grouping for proper weight decay handling.

### 1. Weight Decay
Weight decay is applied to:

- linear weights
- embedding weights

Weight decay is **not** applied to:

- biases
- LayerNorm parameters

### 2. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

This helps keep training stable.

### 3. Learning Rate Scheduler
The schedule has 3 phases:

1. **warmup**: linearly increase learning rate
2. **cosine decay**: smoothly decrease learning rate
3. **minimum LR floor** after decay

---

## Installation

Create and activate a Python environment, then install dependencies.

```bash
pip install torch tiktoken transformers
```

If you also want notebook support:

```bash
pip install jupyter matplotlib
```

---

## How to Run

### For training the model
Run

```bash
python train.py
```

### For generating text
If generation code is enabled in the script, the model can generate text from a prefix such as:

```text
"Hello, I'm a language model, "
```
Run 
```bash
python inspect_model.py
```


---

## Example Training Loop Behavior

During training, the script prints statistics like:

- step number
- loss
- gradient norm
- time per step
- tokens processed per second

Example:

```text
step    0 | loss: 10.923451 | norm: 1.2874 | dt: 82.14ms | tok/sec: 779.1
```

---


## Learning

This repository is intended as a learning project for understanding:

- how GPT models are structured
- how multi-head attention is implemented in code
- how tensors are reshaped during QKV splitting and head merging
- how language modeling loss is formed
- how warmup and cosine decay schedules work
- how weight initialization and residual scaling stabilize training

It is especially useful for **students and enthusiasts** who want to move from using pretrained models to **understanding how they are built internally**.

---

## Acknowledgment
This implementation is heavily inspired by the teaching style and GPT training explanations of **Andrej Karpathy**, especially the minimal and educational style of nanoGPT-like code.

---

## 📄 License
MIT

