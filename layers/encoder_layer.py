# layers/encoder_layer.py
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self Attention
        _src = src
        src = self.norm1(src + self.dropout1(self.self_attn(src, src, src, src_mask)))
        # Feed Forward
        src = self.norm2(src + self.dropout2(self.ff(src)))
        return src
