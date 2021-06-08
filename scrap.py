import torch


# mask = torch.arange(20).resize(4,5)
# chosen_nodes = torch.Tensor([[1], [4], [3], [2]])
# print(mask)
# print(chosen_nodes)
#


A = torch.arange(5, dtype=torch.float32, requires_grad=True)
B = torch.zeros(size=(3,5), dtype=torch.float32, requires_grad=True)
F = torch.sin(A**2 + A)

B = 
print(F)
