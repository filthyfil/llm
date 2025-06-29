import torch
import tiktoken
from config import GPT_CONFIG_124M
from model.model import GPTModel
from utils.generate import generate_text, generate_text_with_cache

def main():
    with open("assets/text.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(raw_text)
    input_tensor = torch.tensor(token_ids).unsqueeze(0)

    torch.manual_seed(42)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output = generate_text_with_cache(
        model=model,
        idx=input_tensor,
        max_new_tokens=50,
        context_length=GPT_CONFIG_124M["context_length"]
    )

    decoded = tokenizer.decode(output.squeeze(0).tolist())
    print(decoded)

if __name__ == "__main__":
    main()
