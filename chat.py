import torch
import tiktoken
from config import GPT_CONFIG_124M
from model.model import GPTModel
from utils.generate import chat

def main():
    # Load system prompt
    with open("assets/system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(42)

    # Model selection and config
    model_name = "gpt2-small (124M)"
    model_path = "gpt2-small_(124M).pth"

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

    model = GPTModel(NEW_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print(f"🧠 GPT-2 model loaded on {device}")
    print("Type 'exit' to quit.\n")

    # Chat loop
    chat_history = system_prompt
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        chat_history += f"\nYou: {user_input}\nAI:"
        tokens = tokenizer.encode(chat_history)

        # Truncate tokens if necessary
        if len(tokens) > 1024:
            tokens = tokens[-1024:]

        input_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

        output = chat(
            model=model,
            idx=input_tensor,
            max_new_tokens=100,
            context_size=1024,
            top_k=25,
            temperature=1.0,
            use_cache=False,
            end_token_id=50256  # GPT-2's end-of-text token
        )

        full_text = tokenizer.decode(output.squeeze(0).tolist())
        response = full_text.split("AI:")[-1].strip().split("\n")[0]
        print(f"AI: {response}")
        chat_history += f" {response}"

if __name__ == "__main__":
    main()
