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
├── gpt2/                  # Saves raw model parameters and settings 
├── model/
│   ├── gpt_model.py       # GPTModel class
│   ├── transformer_block.py  # TransformerBlock, LayerNorm, FeedForward
│   ├── attention.py       # MultiHeadAttention, CausalSelfAttention
│   └── activations.py     # GELU activation
├── utils/
│   ├── generate.py        # Text generation function
│   ├── gpt_download.py    # Set up function, grabs weights online
│   └── load_weights.py    # Loads weights into the model
├── README.md              # You are here!
├── config.py              # Model configuration dictionary
├── main.py                # Entry point for text generation
├── requirements.txt       # Dependencies
└── setup.py               # Setup script 
```

## 🧪 Quickstart

1. **Install dependencies**  
Run this command (in a virtual environment):
```bash
pip install -r requirements.txt
```

2. **Install the model**
For testing, install the gpt-small (124M) model.
```bash
python setup.py
```

3. **Run the model**
On the gpt2-small (124M) model the output should read: "The meaning of life is the end. This is not the end of us — 
you're never alone" on the input "The meaning of life is".
```bash
python main.py
```
4. **Change the prompt**
You can change the prompt in assets/text.txt

The model will read `text.txt`, tokenize it, and generate text token-by-token using causal self-attention.
On the gpt2-small (124M) model it should read: "The meaning of life is the end. This is not the end of us — 
you're never alone" on input "The meaning of life is".

---

## ⚙️ Configuration

Edit `config.py` to modify model hyperparameters (Only do this if you know what you are doing!):
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
