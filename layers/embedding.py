import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        # x: [batch_size, seq_len]
        # Scale the embedding by sqrt of the embedding for better initialisation (according to the paper)
        return self.embedding(x) * (self.embedding.embedding_dim ** 0.5)

    