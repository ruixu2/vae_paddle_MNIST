import torch
import torch.nn as nn

# example input and target
input = torch.Tensor([[1, 2, 3], [4, 5, 6]])
target = torch.Tensor([[2, 2, 2], [6, 6, 6]])

# create KLDivLoss module
KLDivLoss = nn.KLDivLoss(reduction='batchmean')

# compute KL divergence
log_input = nn.functional.softmax(input, dim=1)
prob_target = nn.functional.softmax(target, dim=1)
KLD = KLDivLoss(log_input, prob_target)
print(log_input)
print(prob_target)
print(KLD)