import torch
from lenet_class import LeNet

model = LeNet()

inp = torch.rand(32,32)
inp = inp.reshape(1,32,32)

output = torch.nn.Softmax(dim=1)(model(inp))
print(output)