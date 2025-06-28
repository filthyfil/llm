import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "dim_embed": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


# GPTModel is a simple implementation of the GPT architecture
# It consists of an embedding layer, multiple transformer blocks, and a final output head.
# The embedding layer converts input token indices into dense vectors,
# the transformer blocks process these embeddings through self-attention and feed-forward networks,
# and the output head produces logits for each token in the vocabulary.    
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg["vocab_size"], cfg["dim_embed"])
        self.pos_embed = nn.Embedding(cfg["context_length"], cfg["dim_embed"])  
        self.drop_embed = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["dim_embed"])
        self.out_head = nn.Linear(
            cfg["dim_embed"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embed = self.tok_embed(in_idx)
        pos_embed = self.pos_embed(torch.arange(seq_len, device=in_idx.device))
        
        x = tok_embed + pos_embed
        x = self.drop_embed(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        return logits



# TransformerBlock is a single block of the transformer architecture
# It consists of a multi-head self-attention layer followed by a feed-forward network.
# The reason it is this specific structure is to allow the model to learn complex relationships
# between the input tokens while also being able to process them in parallel.
# The attention mechanism allows the model to focus on different parts of the input sequence,
# while the feed-forward network allows it to learn complex transformations of the input.
# The residual connections and layer normalization help stabilize the training process.
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg["dim_embed"],
            d_out=cfg["dim_embed"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
            dropout=cfg["drop_rate"]
        )
        self.feed_forward = FeedForward(cfg)
        self.layer_norm1 = LayerNorm(cfg["dim_embed"])
        self.layer_norm2 = LayerNorm(cfg["dim_embed"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x # <residual connection
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # residual connection/>

        shortcut = x # <residual connection
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # residual connection/>

        return x
    

# Layer normalization is a technique to normalize the inputs across the features
# It helps stabilize the learning process and can lead to faster convergence.
class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# GELU activation function as described in the original GPT paper
# It is a smooth approximation of the ReLU activation function.
# The GELU activation function is defined as below:
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))))
    

# Feed forward is a simple MLP with GELU activation
# It takes the input, applies a linear transformation, then GELU activation,
# and finally another linear transformation to produce the output.
# The first linear layer expands the dimension to 4 times the original dimension,
# and the second linear layer reduces it back to the original dimension.
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["dim_embed"], 4 * cfg["dim_embed"]),
            GELU(),
            nn.Linear(4 * cfg["dim_embed"], cfg["dim_embed"])
        )
    def forward(self, x):
        return self.layers(x)


# GPTDataset is a custom dataset class for the GPT model
# It takes a text input, tokenizes it using the GPT tokenizer,
# and creates input-target pairs for training.
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length] 
            target_chunk = token_ids[i + 1:i + max_length + 1]
            # `:` is a slice operator, here it takes a slice of the list and returns a new sublist from 
            # i to i + max_length
            # i + max_length is exclusive, so it will not include the token at that index
            # e.g. if i = 0 and max_length = 4, input_chunk will be token_ids[0:4] which is the first 4 tokens
            # if i = 1 and max_length = 4, input_chunk will be token_ids[1:5] which is the second to fifth tokens
            # this is how we create the input chunk

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            # when we append the input and target chunks, we convert them to tensors of type long
            # adding a new tensor as an entry in a list of training samples

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# CausalSelfAttention is a self-attention mechanism that only attends to previous tokens in the sequence.
class CausalSelfAttention(nn.Module):
    # B : batch size
    # T : number of tokens in the sequence
    # D_in : input dimension
    # D_out : output dimension
    # context_length : maximum number of tokens in the sequence
    # qkv_bias : whether to use bias in the linear layers for query, key, and value
    # dropout : dropout rate for attention weights
    # mask : upper triangular mask to prevent attending to future tokens
    def __init__(self, d_in, d_out, context_length, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) 
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch_size, number_of_tokens, d_in = x.shape # x.shape = (B, T, D_in)
        queries = self.W_query(x) # shape: (B, T, D_out)
        keys = self.W_key(x) # shape: (B, T, D_out)
        values = self.W_value(x) # shape: (B, T, D_out)

        attention_scores = queries @ keys.transpose(1, 2) # shape: (B, T, T)
        attention_scores.masked_fill_(self.mask.bool()[:number_of_tokens, :number_of_tokens], -torch.inf) # _ ops are in-place operations

        attention_weights = torch.softmax(attention_scores / torch.sqrt(torch.tensor(keys.shape[-1])), dim=-1) # shape: (B, T, T)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values # shape: (B, T, D_out)
        return context_vector


# MultiHeadAttentionWrapper is a wrapper for multiple heads of self-attention.
# It uses the CausalSelfAttention class to create multiple heads,
# concatenates their outputs, and applies a final linear projection.
class MultiHeadAttentionWrapper(nn.Module): 
    def __init__(self, d_in, d_out, context_length, num_heads=1, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalSelfAttention(d_in, d_out, context_length, qkv_bias, dropout) 
            for _ in range(num_heads)]
        )
        self.out_projection = nn.Linear(d_out*num_heads, d_out*num_heads)

    def forward(self, x):
        context_vector = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.out_projection(context_vector)  # shape: (B, T, D_out*num_heads)


# MultiHeadAttention is a stand-alone multi-head attention mechanism. 
# It splits the input into multiple heads, applies self-attention to each head.
class MultiHeadAttention(nn.Module):
    """
    torch.split(tensor, split_size, dim=0)  
        - splits the tensor into chunks
        - https://docs.pytorch.org/docs/stable/generated/torch.split.html
    torch.chunk(input, chunks, dim=0)
        - chunks is int and could be the number of heads
        - avoids copying tensors, less memory overhead
        - https://docs.pytorch.org/docs/stable/generated/torch.chunk.html
    torch.view
        - best option
        - used to reshape tensors without changing their data. It allows 
          you to specify a new shape for the tensor, as long as the total 
          number of elements remains the same. 
        - https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
    """
    def __init__(self, d_in, d_out, context_length, num_heads=1, qkv_bias=False, dropout=0.0):
        super().__init__()
        assert (d_out % num_heads == 0) 

        self.d_in = d_in
        self.d_out = d_out
        self.d_head = d_out // num_heads
        self.num_heads = num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) 
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch_size, number_of_tokens, d_in = x.shape # x.shape = (B, T, D_in)
        queries = self.W_query(x) # shape: (B, T, D_out)
        keys = self.W_key(x) # shap context_vector = (attention_weights @ values).transpose(1, 2) 
        # shape: e: (B, T, D_out)
        values = self.W_value(x) # shape: (B, T, D_out)

        # partition the qk tensors for each head (mh)
        # mh_queries = torch.chunk(queries, num_heads, dim=-1)
        # mh_keys = torch.chunk(keys, num_heads, dim=-1)
        # ^ this is a bad idea because it is iterative, a better 
        # approach is to define a new dimension in the tensor
        # this is an idea in physics too, where you can hide a
        # density function \int p(x) dx in a dimension               
        
        queries = queries.view(
            batch_size, number_of_tokens, self.num_heads, self.d_head
            ).transpose(1, 2) # shape : (B, T, H, D_head) -> (B, H, T, D_head)
        keys = keys.view(
            batch_size, number_of_tokens, self.num_heads, self.d_head
            ).transpose(1, 2) # shape : (B, T, H, D_head) -> (B, H, T, D_head)
        values = values.view(
            batch_size, number_of_tokens, self.num_heads, self.d_head
            ).transpose(1, 2) # shape : (B, T, H, D_head) -> (B, H, T, D_head)

        # queries = queries.transpose(1, 2) # shape : 
        # keys = keys.transpose(1, 2) # shape : (B, H, T, D_head)
        # values = values.transpose(1, 2) # shape : (B, H, T, D_head)

        attention_scores = queries @ keys.transpose(2, 3)
        mask = self.mask.bool()[:number_of_tokens, :number_of_tokens]
        attention_scores.masked_fill_(mask, -torch.inf)
        attention_weights = torch.softmax(attention_scores / torch.sqrt(torch.tensor(keys.shape[-1])), dim=-1) # shape: (B, T, T)
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1, 2) 
        # shape: (B, T, T, H)
        context_vector = context_vector.contiguous().view(batch_size, number_of_tokens, self.d_out)
        context_vector = self.out_proj(context_vector)

        return context_vector


# create_dataloader is a utility function to create a DataLoader for the GPTDataset.
# It takes a text input, batch size, maximum sequence length, stride, shuffle option,
# whether to drop the last batch, and number of workers for data loading.
# It is used in the training loop to load the data in batches for training the GPT model.
def create_dataloader(txt, batch_size=4, max_length=4, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


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


def main(): 
    with open("text.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_text = tokenizer.encode(raw_text)

    encoded_tensor = torch.tensor(encoded_text).unsqueeze(0)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    out = generate_text(
        model=model,
        idx=encoded_tensor, 
        max_new_tokens=100, 
        context_length=GPT_CONFIG_124M["context_length"]
    )

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)

    return


if __name__ == '__main__':
    main()

