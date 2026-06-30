import math
import torch
from torch import nn
from torch.nn import functional as F


class Router(nn.Module):
  def __init__(self, embedding_dim, num_expert, top_k, user_noisy_top_k=True, capacity_factor=1.25):
    super().__init__()

    self.embedding_dim = embedding_dim
    self.num_expert = num_expert
    self.top_k = top_k
    assert self.top_k >= 1 and self.top_k <= num_expert
    self.use_noisy_top_k = user_noisy_top_k
    self.capacity_factor = capacity_factor
    self.w_g = nn.Linear(embedding_dim, num_expert, bias=False)
    self.w_noise = nn.Linear(embedding_dim, num_expert, bias=False) if self.use_noisy_top_k else None

  def forward(self, x):
    # get the total number of tokens in the batch
    B, C, _ = x.size()
    num_tokens = B * C

    logits = self.w_g(x)
    if self.use_noisy_top_k:
      #(optionally) add noise into the router
      noise = F.softplus(self.w_noise(x))
      noise *= torch.randn_like(noise)
      logits += noise

    # top-k expert selection, compute probilities over actuve experts
    top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
    router_probs = torch.full_like(logits, float('-inf'))
    router_probs.scatter_(-1, top_k_indices, top_k_logits)
    router_probs = F.softmax(router_probs, dim=-1)

    # compute the expert capacity
    exp_capacity = math.floor(self.top_k * self.capacity_factor * num_tokens / self.num_expert)
    exp_capacity += exp_capacity % 2 # make sure expert capacity is an even integer
    exp_capacity = int(exp_capacity)

    # make a multi-hot mask of chosen experts
    # values are 0 if expert not chosen, 1 if expert chosen
    exp_mask = F.one_hot(top_k_indices, num_classes=self.num_expert)
    exp_mask = exp_mask.view(num_tokens, self.top_k, self.num_expert)
    exp_mask = exp_mask.permute(1,0,2)

    # compute index for each token in expert batch
    # NOTE: cumsum counts top-1 first, top-2 second, etc.
    # to prioritize top experts when dropping tokens
    exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.num_expert)
    exp_rank = torch.cumsum(exp_rank, dim=0)
    exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.num_expert)

    # mask entries beyind expert capacity and compute used capacity
    exp_mask *= torch.lt(exp_rank, exp_capacity)

    # matrix storing tokens position in batch of corresponding expert
    exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)

    # mask probabilities to only include selected experts
    router_probs = router_probs.view(num_tokens, self.num_expert)[None, :]
    exp_weights = exp_mask * router_probs

    # position of each toekn within the capacity of the selected expert
    exp_rank_sc = F.one_hot(exp_rank, num_classes = exp_capacity)

    # weight of the selected expert for each token at position the capacity of that expert
    exp_weights = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)
    exp_mask = exp_weights.bool()

    # reshape token into batches for each expert, return both waights and batches
    # [num_expert, exp_capacity, B*C] * [B*C, embedding_dim] -> [num_expert, exp_capacity, n_embed]
    x = x.view(num_tokens, self.embedding_dim)
    expert_batches = exp_mask.permute(1,2,0).type_as(x) @ x
    return exp_weights, exp_mask, expert_batches
