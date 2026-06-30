import math
import torch
from torch import nn
import torch.nn.functional as F


class CasualSelfAttention(nn.Module):

  def __init__(self, embedding_dim: int, num_heads: int, max_token_length: int, bias: bool=False, dropout: float=0.0):
    super().__init__()

    assert embedding_dim % num_heads == 0
    #"Embedding dimension must be divisible by the number of heads"

    # key, query, value projections for all heads, but in a batch
    # Output is 3X the dimension because it includes key, query and value
    self.c_attn = nn.Linear(embedding_dim, 3*embedding_dim, bias=bias)

    #projection of concatenated attention head outputs
    self.c_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

    # regularization
    self.attn_dropout = nn.Dropout(dropout)
    self.resid_dropout = nn.Dropout(dropout)
    self.num_heads = num_heads
    self.embedding_dim = embedding_dim

    # causal mask to ensure that attention is only applied to
    # the left in the input sequence
    self.register_buffer("mask", torch.tril(torch.ones(max_token_length, max_token_length)))

  def forward(self, x):
    B, T, C = x.size()
    # batch size, sequence length, embedding dimensionality

    # Compute query, key, and value vectors for all heads in batch
    # split the output into separate query, key, and value tensorsd
    # [B, T, embedding_dim]
    q, k, v = self.c_attn(x).split(self.embedding_dim, dim=2)

    # reshape tensor into sequence of smaller token vectors for each head
    k = k.view(B, T, self.num_heads, self.embedding_dim // self.num_heads).transpose(1,2) # [B, T, num_head, emnedding_dim//num_head]
    q = q.view(B, T, self.num_heads, self.embedding_dim//self.num_heads).transpose(1,2) # [B, T, num_head, emnedding_dim//num_head]
    v = v.view(B, T, self.num_heads, self.embedding_dim//self.num_heads).transpose(1,2) # [B, T, num_head, emnedding_dim//num_head]

    # Compute the attention matrix, perform maskeing, and apply dropout
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)

    # compute output vectors for each token
    y = att @ v # [B, num_head, T, T] x [B, num_head, T, emnedding_dim//num_head] -> [B, num_head, T, emnedding_dim//num_head]

    # Concatenate outputs from each attention head and linearly project
    y = y.transpose(1, 2).contiguous().view(B, T, self.embedding_dim) # [B, num_head, T, emnedding_dim//num_head] -> [B, T, embedding_dim]
    y = self.resid_dropout(self.c_proj(y))
    return y
