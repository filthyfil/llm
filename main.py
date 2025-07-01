import torch
import tiktoken
from config import GPT_CONFIG_124M
from model.model import GPTModel
from utils.generate import generate
# from utils.gpt_download import download_and_load_gpt2
# from utils.load_weights import assign, load_weights_into_gpt


def main():
    with open("assets/text.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(raw_text)
    input_tensor = torch.tensor(token_ids).unsqueeze(0)

    torch.manual_seed(42)

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # Copy the base config (small) and update to specific model settings
    model_name = input(f"Model? (ex: gpt2-small (124M)): ").strip()
    model_path = model_name.replace(" ", "_") + ".pth"
    print(f"Selected model: {model_name}")
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

    model = GPTModel(NEW_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model configuration: {NEW_CONFIG}")
    model
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    input_tensor = input_tensor.to(device)

    model.eval()

    output = generate(
        model=model,
        idx=input_tensor,
        max_new_tokens=15,
        context_size=NEW_CONFIG["context_length"],
        top_k=25,
        temperature=1.4,
        use_cache=False
    )

    decoded = tokenizer.decode(output.squeeze(0).tolist())
    print(decoded)
    print("Generation complete.")
    print("On gpt2-small (124M) it should read: \"The meaning of life is the end. This is not the end of us — you're never alone\"")
    # torch.save(model.state_dict(), "small-gpt2.pth")


if __name__ == "__main__":
    main()