import torch
import torch.nn as nn

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
    

# MultiHeadAttentionWithCaching is a stand-alone multi-head attention mechanism. 
# It splits the input into multiple heads, applies self-attention to each head.
# This implementation has KV caching.
class MultiHeadAttentionWithCaching(nn.Module):
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
    def __init__(self, d_in, d_out, context_length, num_heads=1, qkv_bias=False, dropout=0.0, max_seq_len=None, window_size=None):
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
        # upper triangular mask to prevent attending to future tokens
        # mask.shape = (context_length, context_length)
        self.max_seq_len = max_seq_len or context_length
        self.window_size = window_size or self.max_seq_len
        self.register_buffer("cache_k", None, persistent=False) # kv cache
        self.register_buffer("cache_v", None, persistent=False) # kv cache
        self.ptr_current_pos = 0
        # cache_k and cache_v are used to store the keys and values for the attention mechanism
        # this is useful for inference, where we want to cache the keys and values
        # to avoid recomputing them for each token in the sequence

    def reset_cache(self):
        # reset the cache for keys and values
        self.cache_k = None
        self.cache_v = None
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):
        batch_size, number_of_tokens, d_in = x.shape # x.shape = (B, T, D_in)
        # queries = self.W_query(x) # shape: (B, T, D_out)
        # keys = self.W_key(x) # shape (B, T, D_out)
        # values = self.W_value(x) # shape: (B, T, D_out)

        # kv cache implementation
        keys_new = self.W_key(x)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)

        # partition the qk tensors for each head (mh)
        # mh_queries = torch.chunk(queries, num_heads, dim=-1)
        # mh_keys = torch.chunk(keys, num_heads, dim=-1)
        # ^ this is a bad idea because it is iterative, a better 
        # approach is to define a new dimension in the tensor
        # this is an idea in physics too, where you can hide a
        # density function \int p(x) dx in a dimension      
        keys_new = keys_new.view(batch_size, number_of_tokens, self.num_heads, self.d_head)
        values_new = values_new.view(batch_size, number_of_tokens, self.num_heads, self.d_head)
        queries = queries.view(batch_size, number_of_tokens, self.num_heads, self.d_head)

        if use_cache:
            if self.cache_k is None:
                self.cache_k = keys_new.detach()
                self.cache_v = values_new.detach()
                print("KV-cache is empty.")
            else:
                # Sanity check batch size
                assert self.cache_k.shape[0] == keys_new.shape[0], "Batch size mismatch in cache."
                self.cache_k = torch.cat([self.cache_k, keys_new.detach()], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new.detach()], dim=1)
                print("KV-cache cat.")
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new
            print("KV-cache is not used.")

                 

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # queries = queries.transpose(1, 2) # shape : 
        # keys = keys.transpose(1, 2) # shape : (B, H, T, D_head)
        # values = values.transpose(1, 2) # shape : (B, H, T, D_head)

        # mask = self.mask.bool()[:number_of_tokens, :number_of_tokens]
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        # In KV caching, the number of tokens becomes ambiguous
        # because the query comes from only the new input tokens,
        # while the keys and values may include all previously cached tokens.
        # So we must separately track:
        #   - the number of query tokens (new tokens just fed in)
        #   - the number of key/value tokens (all tokens seen so far, due to caching)
        # If caching is enabled, we also need to use a pointer (e.g. self.ptr_current_pos)
        # to extract the correct portion of the attention mask for this step.
        if use_cache:
            mask_bool = self.mask.bool()[
                self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
            ]
            self.ptr_current_pos += num_tokens_Q
        else:
            mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]

        # Now, the attention mechanism can be computed
        attention_scores = queries @ keys.transpose(2, 3) # shape: (B, H, T, T)
        attention_scores.masked_fill_(mask_bool, -torch.inf)
        attention_weights = torch.softmax(attention_scores / (keys.shape[-1] ** 0.5), dim=-1) # shape: (B, H, T, T)
        attention_weights = self.dropout(attention_weights)
        
        context_vector = (attention_weights @ values).transpose(1, 2) # shape: (B, H, T, D_head) -> (B, T, H, D_head)
        context_vector = context_vector.contiguous().view(batch_size, number_of_tokens, self.d_out) # shape: (B, T, D_out)
        context_vector = self.out_proj(context_vector)

        print(f"Cache K shape: {self.cache_k.shape}")

        

        return context_vector
    