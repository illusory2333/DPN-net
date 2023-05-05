import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

class DFB(nn.Module):
    def __init__(self):
        super(DFB, self).__init__()
        self.conv1 = nn.Conv2d(2048, 512, 1)
        self.dwconv = nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=512)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(512, 2048, 1)
        self.bn1 = nn.BatchNorm2d(512)
    def forward(self, x):
        identify = x
        x = self.conv1(x)
        x1 = F.relu(self.bn1(self.conv3(x)))
        x2 = x1 + x
        x2 = F.relu(self.bn1(self.dwconv(x2)))
        x2 = x2 + x
        x2 = F.relu(self.bn1(self.dwconv(x2)))
        output = x1 + x2
        output = self.conv11(output)
        out = output + identify
        return out
if __name__ == '__main__':
    x = torch.randn(4, 2048, 8, 6)
    model = DFB()
    x = model(x)
    print(x.shape)



