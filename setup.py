import torch
import re
import os
from config import GPT_CONFIG_124M
from tabulate import tabulate
from model.model import GPTModel
from utils.gpt_download import download_and_load_gpt2
from utils.load_weights import load_weights_into_gpt

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Display available models
table = [[name, *cfg.values()] for name, cfg in model_configs.items()]
headers = ["NAME", "Embedding Dim", "Layers", "Heads"]
print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

# Prompt for model selection (2 attempts max)
for _ in range(2):
    choice = input("\nType the model name you want to install (e.g., gpt2-small (124M)): ").strip()
    if choice in model_configs:
        print(f"\n✅ You selected: {choice}")
        break
    print(f"\n❌ Invalid choice: {choice}")
else:
    raise ValueError("Too many invalid attempts. Exiting.")

# Extract model size and save path
match = re.search(r"\((.*?)\)", choice)
if not match:
    raise ValueError(f"Could not extract model size from '{choice}'")
model_size = match.group(1)
model_save_path = f"{choice}.pth".replace(" ", "_")

# Load or download and save weights
if not os.path.exists(model_save_path):
    print("📥 Downloading weights...")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    config = GPT_CONFIG_124M.copy()
    config.update(model_configs[choice])
    config.update({"context_length": 1024, "qkv_bias": True})

    model = GPTModel(config)
    load_weights_into_gpt(model, params)

    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model weights saved to {model_save_path}")
else:
    print(f"✅ Model weights already exist at {model_save_path}")
