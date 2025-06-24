import tiktoken
import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class testGPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        batch_size, seq_len, _ = attention_scores.size()
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = causal_mask.masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, 0.0)
        print(mask)
        attention_scores = attention_scores + mask

        attention_weights = torch.softmax(attention_scores / torch.sqrt(torch.tensor(keys.shape[-1], dtype=torch.float32)), dim=-1)
        print(attention_weights)
        context_vector = attention_weights @ values

        return context_vector
        

def test_create_dataloader(txt, batch_size=4, max_length=4, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = testGPTDataset(txt, tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader
# will use custom cuda kernel for optimizationnnnn laterrrr

def main(): 
    # text = "Hello, world . This is a TESTTTTT !!! lol"
    # result = re.split(r'([,.]|\s)', text)
    # result = [item for item in result if item.strip()]
    # print(result)

    tokenizer = tiktoken.get_encoding("gpt2")
    with open("text.txt", "r", encoding="utf-8") as file:
        raw_text = file.read()

    # print(raw_text)

    enc_text = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
    print(f"Text data length: {len(enc_text)}")
    # enc_sample = enc_text[50:]
    # # print(enc_sample)

    # context_frame = 4
    # x = enc_sample[:context_frame]
    # y = enc_sample[:context_frame + 1]
    # print(f"Looking: {x}")
    # print(f"Predicting: {y}")

    # for i in range(1,context_frame+1):
    #     context = enc_sample[:i]
    #     desired = enc_sample[i]
    #     print(tokenizer.decode(context), "->", tokenizer.decode([desired]))
    # with open("text.txt", "r", encoding="utf-8") as f:
    #     raw_text = f.read()

    max_length = 4
    dataloader = test_create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=4, shuffle=False, drop_last=False, num_workers=0)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("INPUTS:\n", inputs)
    print("\nTARGETS:\n", targets)

    torch.manual_seed(123)
    vocab_size = tokenizer.n_vocab # gpt2 vocab size
    print(f"VOCAB SIZE: {vocab_size}")
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)

    # print(token_embeddings)
    # abs embedded approach as in gpt2
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    input_embeddings = token_embeddings + pos_embeddings
    # print(input_embeddings)

    # query = inputs[1]
    # attention_scores_2 = torch.empty(inputs.shape[0])
    # for i, x_i in enumerate(inputs):
    #     attention_scores_2[i] = torch.dot(x_i, query)
    # attention_weights_2 = torch.softmax(attention_scores_2, dim=0)
    # print(f"Normalized attention scores: {attention_weights_2}")

    # context_vector_2 = torch.zeros(query.shape)
    # for i, x_i in enumerate(inputs):
    #     context_vector_2 += attention_weights_2 * x_i
    # print(f"Context vector: {context_vector_2}")

    # attention_scores = inputs @ inputs.T
    # print(attention_scores)
    # attention_weights = torch.softmax(attention_scores, dim=-1, dtype='utf-8')
    # print(attention_weights)

    # x_2 = input_embeddings[1]
    # d_in = input_embeddings.shape[1]
    # d_out = 2

    # W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
    # W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
    # W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

    # query_2 = x_2 @ W_query
    # key_2 = x_2 @ W_key
    # value_2 = x_2 @ W_value

    # keys = inputs @ W_key
    # values = inputs @ W_value
    # print(f"key shape: {keys.shape}")
    # print(f"value shape: {values.shape}")

    print(input_embeddings.shape)
    self_attention = SelfAttention(output_dim, output_dim)
    print(self_attention(input_embeddings))

    

    return

if __name__ == '__main__':
    main()

