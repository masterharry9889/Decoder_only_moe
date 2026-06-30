import math
import torch
from torch import nn
from torch.nn import functional as F
from router import Router
from experts_layer import MLPExperts
from attention_mecanism import CasualSelfAttention
class MOELayer(nn.Module):
  def __init__(self, embedding_dim, num_expert, top_k, use_noisy_top_k=True, capacity_factor=1.25, bias=False, dropout=0.2):
    super().__init__()

    self.router = Router(embedding_dim,
                         num_expert,
                         top_k,
                         use_noisy_top_k,
                         capacity_factor)
    self.experts = MLPExperts(embedding_dim=embedding_dim,
                              num_expert=num_expert,
                              bias=bias,
                              dropout=dropout)

  def forward(self, x: torch.Tensor):
    B, C, d = x.size()  # track the original shape of input
    num_tokens = (B*C)

    # pass each token through the router
    exp_weight, exp_mask, exp_batches = self.router(x)

    # compute expert outout
    exp_out = self.experts(exp_batches)

    # aggregate expert output based on router weights
    exp_weigth = exp_weight.view(num_tokens, -1)
    exp_out = exp_out.view(-1, d)
    output = exp_weigth @ exp_out

    return output.view(B, C, d)
