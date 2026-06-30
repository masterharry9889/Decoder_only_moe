import torch
import torch.nn.functional as F


Batch_size = 16
sequence_length = 256
num_expert = 8
k = 2  # number of active expert


indices = torch.randint(0, num_expert, (Batch_size, sequence_length, k))
expert_probs = F.softmax(torch.rand(Batch_size, sequence_length, num_expert), dim=2)

with torch.no_grad():
  one_hot_indices = F.one_hot(indices, num_classes=num_expert)
  one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)
  tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0,1))

prob_per_expert = torch.mean(expert_probs.float(), dim=(0,1))

load_balance_loss = num_expert * torch.sum(prob_per_expert * tokens_per_expert)
print(f"Load Balance Loss: {load_balance_loss}")
