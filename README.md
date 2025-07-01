# 🧠 GPT-from-Scratch

A minimal, readable implementation of a GPT-style language model built from scratch in PyTorch.  
This project is designed for learning, experimentation, and modification — not production deployment.

Shout out to Sebastian Raschka for his work in writing "Build a Large Language Model (From Scratch)"

## 🚀 Features

- Implements the GPT architecture (embedding → transformer blocks → output head)
- Multi-head self-attention with causal masking
- Configurable number of layers, heads, embedding dimensions
- Modular design (easily extendable to add KV cache, training loop, etc.)
- Tokenization using OpenAI's `tiktoken`
- Text generation from prompt

## 📁 Project Structure

```
gpt_project/
├── assets/
│   ├── metamorphosis.txt  # Cool book
│   └── text.txt           # Text generation function
├── data/
│   ├── dataset.py         # GPTDataset (tokenization + chunking)
│   └── loader.py          # DataLoader 
├── model/
│   ├── gpt_model.py       # GPTModel class
│   ├── transformer_block.py  # TransformerBlock, LayerNorm, FeedForward
│   ├── attention.py       # MultiHeadAttention, CausalSelfAttention
│   └── activations.py     # GELU activation
├── utils/
│   └── generate.py        # Text generation function
├── README.md              # You are here!
├── config.py              # Model configuration dictionary
├── main.py                # Entry point for text generation
└── requirements.txt       # Dependencies
```

## 🧪 Quickstart

1. **Install dependencies**  
Just pytorch and tiktoken; to install my specific versions:
```bash
pip install -r requirements.txt
```

2. **Add your prompt** to `text.txt`

3. **Run the model**
```bash
python main.py
```

The model will read `text.txt`, tokenize it, and generate text token-by-token using causal self-attention.

---

## ⚙️ Configuration

Edit `config.py` to modify model hyperparameters:
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "dim_embed": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

---

## 📚 Future Additions

- [ ] Training loop (AdamW, LR warmup, etc.)
- [x] KV cache for faster inference
- [ ] Weight initialization strategies
- [ ] Model checkpointing and logging
- [ ] CoT

---

## 🧠 Why This Exists

Most open-source GPT repos are large and difficult to read.  
This project is meant to be a **minimalist, hackable reference** for understanding how GPT works under the hood.

---
