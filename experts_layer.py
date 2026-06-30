import torch
from torch import nn


class MLPExperts(nn.Module):

  def __init__(self, embedding_dim, num_expert, bias=False, dropout=0.2):
    super().__init__()

    self.bias = bias
    self.c_fc = nn.Parameter(torch.empty(num_expert, embedding_dim, 4*embedding_dim))
    self.c_proj = nn.Parameter(torch.empty(num_expert, 4*embedding_dim, embedding_dim))
    self.fc_bias = nn.Parameter(torch.empty(num_expert, 1, 4*embedding_dim)) if self.bias else None
    self.proj_bias = nn.Parameter(torch.empty(num_expert, 1, embedding_dim)) if self.bias else None
    self.gelu = nn.GELU()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = torch.bmm(x, self.c_fc)
    if self.bias:
      x += self.fc_bias
    x = self.gelu(x)
    x = torch.bmm(x, self.c_proj)
    if self.bias:
      x += self.proj_bias
    x = self.dropout(x)
    return x
