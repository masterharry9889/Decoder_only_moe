import torch
from torch import nn
from attention_mecanism import CasualSelfAttention
from moe_layer import MOELayer
class MOEBlock(nn.Module):
  def __init__(self, embedding_dim, num_head, max_sequence_length, num_expert, top_k, use_noisy_top_k=True, capacity_factor=1.25, bias=False, dropout=0.2):
    super().__init__()

    self.ln_1 = nn.LayerNorm(embedding_dim)
    self.attn = CasualSelfAttention(embedding_dim=embedding_dim,
                                    num_heads=num_head,
                                    max_token_length=max_sequence_length,
                                    bias = bias,
                                    dropout = dropout)
    self.ln_2 = nn.LayerNorm(embedding_dim)
    self.moe_layer = MOELayer(embedding_dim=embedding_dim,
                              num_expert=num_expert,
                              top_k=top_k,
                              use_noisy_top_k=use_noisy_top_k,
                              capacity_factor=capacity_factor,
                              bias=bias,
                              dropout=dropout)

  def forward(self, x):
    x = x+self.attn(self.ln_1(x))
    x = x+self.moe_layer(self.ln_2(x))
    return x
