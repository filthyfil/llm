import tiktoken
import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

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
        
class MultiHeadAttentionWrapper(nn.Module): # wrapper for multiple heads, can write as a single class with weight splits
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

class MultiHeadAttention(nn.Module):
    """
    research:
    torch.split(tensor, split_size, dim=0) : 
        - splits the tensor into chunks
        - https://docs.pytorch.org/docs/stable/generated/torch.split.html
    torch.chunk(input, chunks, dim=0)
        - chunks is int and could be the number of heads
        - avoids copying tensors, less memory overhead
        - https://docs.pytorch.org/docs/stable/generated/torch.chunk.html
    
    right now, chunk() seems like the better fit, but which is faster    
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
        context_vector = self.out_proj(context_vector)

        return context_vector

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

def main(): 
    with open("text.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_text = tokenizer.encode(raw_text)

    vocab_size = tokenizer.n_vocab  # gpt2 vocab size
    output_dim = 256
    max_len = 1024
    context_length = max_len


    token_embedding_layer = nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = nn.Embedding(context_length, output_dim)

    max_length = 4
    dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length)

    for batch in dataloader:
        x, y = batch

        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        input_embeddings = token_embeddings + pos_embeddings

        break

    print(f"Input embeddings shape: {input_embeddings.shape}")

    attention_layer = MultiHeadAttentionWrapper(
        d_in=output_dim,
        d_out=output_dim,
        context_length=max_length,
        num_heads=4,
        qkv_bias=True,
        dropout=0.1
    )

    attention_output = attention_layer(input_embeddings)
    print(f"Attention output shape: {attention_output.shape}")
    print(f"Attention output: {attention_output}")

    return

if __name__ == '__main__':
    main()

