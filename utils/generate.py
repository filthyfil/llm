import torch


# generate_text is a function to generate text using the GPT model.
# It takes the model, input indices, maximum number of new tokens to generate,
# and the context length as parameters.
# It uses the model to predict the next token based on the input indices,
# appends the predicted token to the input sequence, and repeats this process
# until the desired number of new tokens is generated.
# It also supports caching of key-value pairs for faster generation.
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=50256, use_cache=False):
    model.eval()
    with torch.no_grad():
        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -context_size:]
                logits = model(idx_cond)
                logits = logits[:, -1, :]

                if top_k is not None:
                    top_logits, _ = torch.topk(logits, top_k)
                    min_val = top_logits[:, -1].unsqueeze(1)
                    logits = torch.where(logits < min_val, float("-inf"), logits)

                if temperature > 0.0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)

                if eos_id is not None and (idx_next == eos_id).all():
                    break

                idx = torch.cat((idx, idx_next), dim=1)
        else:
            # ⚠️ RESET CACHE FIRST
            model.reset_kv_cache()
            model.current_pos = 0

            # Initialize cache by processing full prompt once
            logits = model(idx, use_cache=True)

            for _ in range(max_new_tokens):
                logits = logits[:, -1, :]
                if top_k is not None:
                    top_logits, _ = torch.topk(logits, top_k)
                    min_val = top_logits[:, -1].unsqueeze(1)
                    logits = torch.where(logits < min_val, float("-inf"), logits)

                if temperature > 0.0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_idx = torch.multinomial(probs, num_samples=1)
                else:
                    next_idx = torch.argmax(logits, dim=-1, keepdim=True)

                if eos_id is not None and (next_idx == eos_id).all():
                    break

                idx = torch.cat((idx, next_idx), dim=1)

                # ✅ THIS IS KEY: feed only next token, but KV cache holds history
                logits = model(next_idx, use_cache=True)

    return idx


def chat(
    model,
    idx,
    max_new_tokens,
    context_size,
    top_k=None,
    temperature=1.0,
    end_token_id=50256,  # default for GPT-2
):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k:
                top_k_values, _ = torch.topk(logits, top_k)
                logits[logits < top_k_values[:, [-1]]] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Stop generation if end-of-text token is generated
            if idx_next.item() == end_token_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

    return idx
