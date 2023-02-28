import torch

a = torch.tensor([2])
b = torch.tensor([1])
result = not ((a==torch.tensor([1])) ^ (b==torch.tensor([2])))
print(result)