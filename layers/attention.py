# layers/attention.py
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v: [batch_size, seq_len, embed_dim]
        batch_size = q.size(0)

        # Linear projections
        Q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k) # [B, L, H, Dk]
        K = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k)

        # Transpose to [B, H, L, Dk]
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5) # [B, H, L, L]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ V # [B, H, L, Dk]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        out = self.out(out)
        return out
