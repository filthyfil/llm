import torch

# generate_text is a function to generate text using the GPT model.
# It takes the model, input indices, maximum number of new tokens to generate,
# and the context length as parameters.
# It uses the model to predict the next token based on the input indices,
# appends the predicted token to the input sequence, and repeats this process
# until the desired number of new tokens is generated.
def generate_text(model, idx, max_new_tokens, context_length):
    for _ in range(max_new_tokens):
        idx_conditioned = idx[:, -context_length:] 
        with torch.no_grad():
            logits = model(idx_conditioned)
        logits = logits[:, -1, :] # get the last token's logits
        probabilities = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probabilities, num_samples=1) # sample from the distribution
        idx = torch.cat((idx, idx_next), dim=1) # append the new token to the input sequence
    return idx

# generate_text_cached is a function to generate text using the GPT model.
# It takes the model, input indices, maximum number of new tokens to generate,
# and the context length as parameters.
# It uses the model to predict the next token based on the input indices,
# appends the predicted token to the input sequence, and repeats this process
# until the desired number of new tokens is generated.
# It also supports caching of key-value pairs for faster generation.
def generate_text_with_cache(model, idx, max_new_tokens, context_length=None, use_cache=True):
    model.eval()  
    context_len = context_length or model.pos_emb.num_embeddings
    with torch.no_grad():
        if use_cache:
            model.reset_kv_cache()
            logits = model(idx[:, -context_len:], use_cache=True)
            for _ in range(max_new_tokens):
                idx_next = logits[:, -1].argmax(dim=-1, keepdim=True) 
                idx = torch.cat((idx, idx_next), dim=1) 
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -context_len:], use_cache=False)
                idx_next = logits[:, -1].argmax(dim=-1, keepdim=True) 
                idx = torch.cat((idx, idx_next), dim=1)
    return idx

# CODE HERE