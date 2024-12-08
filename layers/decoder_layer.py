# layers/decoder_layer.py
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.ff = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Masked Self-Attention
        _tgt = tgt
        tgt = self.norm1(tgt + self.dropout1(self.self_attn(tgt, tgt, tgt, tgt_mask)))
        # Cross-Attention
        _tgt = tgt
        tgt = self.norm2(tgt + self.dropout2(self.cross_attn(tgt, memory, memory, memory_mask)))
        # Feed Forward
        tgt = self.norm3(tgt + self.dropout3(self.ff(tgt)))
        return tgt
