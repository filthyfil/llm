import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.activations import GELU

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