# models/transformer.py
import torch
import torch.nn as nn
from layers.encoder_layer import EncoderLayer
from layers.decoder_layer import DecoderLayer
from layers.embedding import TokenEmbedding
from layers.positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int = 512,
                 num_heads: int = 8, ff_dim: int = 2048, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.src_embedding = TokenEmbedding(src_vocab_size, embed_dim)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
                                             for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, ff_dim, dropout)
                                             for _ in range(num_decoder_layers)])

        self.linear_out = nn.Linear(embed_dim, tgt_vocab_size)

    def encode(self, src, src_mask):
        x = self.pos_enc(self.src_embedding(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, tgt_mask, memory_mask):
        x = self.pos_enc(self.tgt_embedding(tgt))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask=None):
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, memory_mask)
        out = self.linear_out(out)
        return out
