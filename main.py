import torch
import tiktoken
from config import GPT_CONFIG_124M
from model.model import GPTModel
from utils.generate import generate_text

def main():
    with open("text.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(raw_text)
    input_tensor = torch.tensor(token_ids).unsqueeze(0)

    torch.manual_seed(42)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    output = generate_text(
        model=model,
        idx=input_tensor,
        max_new_tokens=10,
        context_length=GPT_CONFIG_124M["context_length"]
    )

    decoded = tokenizer.decode(output.squeeze(0).tolist())
    print(decoded)

if __name__ == "__main__":
    main()
