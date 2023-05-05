import torch.nn as nn
import  torch
a = torch.randn(3, 96, 96)
down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
a = down(a)
print(a.shape)