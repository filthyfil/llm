import torch
import torch.nn as nn
from model.transformer_block import TransformerBlock, LayerNorm


# GPTModel is a simple implementation of the GPT architecture
# It consists of an embedding layer, multiple transformer blocks, and a final output head.
# The embedding layer converts input token indices into dense vectors,
# the transformer blocks process these embeddings through self-attention and feed-forward networks,
# and the output head produces logits for each token in the vocabulary.    
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["dim_embed"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["dim_embed"])  
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
            )
        self.current_pos = 0

        self.final_norm = LayerNorm(cfg["dim_embed"])
        self.out_head = nn.Linear(
            cfg["dim_embed"], cfg["vocab_size"], bias=False
        )

    def reset_kv_cache(self):
        for block in self.trf_blocks:
            block.att.reset_cache()
        self.current_pos = 0

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        x = self.final_norm(x)

        print(f"Current pos: {self.current_pos}")
        print(f"Input pos_ids: {pos_ids}")

        logits = self.out_head(x)
        return logits