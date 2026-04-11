# Cynaptics-induction_task - GPT-2 Mini (Character-Level Transformer)

## Overview

This project implements a small GPT-2 style Transformer model trained on the Tiny Shakespeare dataset. The model learns character-level language modeling and can generate Shakespeare-like text.

---

## Setup

### 1. Install Dependencies

---

## Dataset

The dataset is automatically downloaded using the provided script:

```bash
python gpt2.py
```

This downloads `shakespeare.txt` into your working directory.

---

## Training (Pretraining Script)

To train the model:

```bash
python gpt2.py
```

What happens:

* Dataset is loaded and tokenized
* Model is initialized
* Training runs for 5000 steps
* Training and validation loss are printed every 500 steps

---

## Text Generation / Autocomplete

After training completes, the script automatically generates text.

Output is saved in:

```
output1.txt
```

You can modify generation length here:

```python
generate(model, context, max_new_tokens=2000)
```

---

## Sample Generated Text

Example output from the model:

```
MENENIIO:
But saill not a like at me thing in the maded,
And The happider' she sond I shonour so the as tall.


ROMEO:
Edwee thank these, do the sin she pulce.

LARICK:
What sis, unceard, my age thee me conces.


MENES:
The town you you dother poour lafter oftnery, they, if the peaterue
Thant his as fane your begett what thy lateree heart' chirder him.
```

---

## Model Architecture

This is a **decoder-only Transformer (GPT-style)** with:

* **Embedding size (n_embd):** 64
* **Number of layers (n_layer):** 4
* **Number of attention heads (n_head):** 8
* **Block size (context length):** 64
* **Dropout:** 0.2

### Key Components

* Token Embeddings + Positional Embeddings
* Multi-Head Self Attention (causal masking)
* Feedforward Network (MLP)
* Residual connections + LayerNorm
* Final linear layer for next-token prediction

---

## Training Details

* Optimizer: AdamW
* Learning Rate: 3e-4
* Batch Size: 32
* Gradient Clipping: 1.0
* Loss: Cross-Entropy

---

## Possible Improvements

* Increase layers and embedding size
* Use GELU instead of ReLU
* Train longer (10k–50k steps)
* Switch to subword tokenization (BPE)
* Add checkpoint saving/loading

---

## File Structure

```
.
├── shakespeare.txt
├── output.txt
└── gpt2.py
```



