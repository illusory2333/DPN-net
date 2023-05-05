import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from networks.CA import CoordAtt

class CANet(nn.Module):
    def __init__(self, channel_settings):
        super(CANet, self).__init__()
        # channel_settings = [2048, 1024, 512, 256]
        self.channel_settings = channel_settings
        laterals = []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
        self.laterals = nn.ModuleList(laterals)
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, input_size,
            kernel_size=1, stride=1, bias=False))
        layers.append(CoordAtt(input_size, input_size))
        layers.append(nn.BatchNorm2d(input_size))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        global_outs = []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[3-i](x[3-i])
            else:
                feature = self.laterals[3-i](x[3-i]) + up
            if i != len(self.channel_settings) - 1:
                BN = nn.BatchNorm2d(self.channel_settings[3-i])
                BN = BN.cuda()
                up = BN(self.down(feature))
                change = nn.Conv2d(in_channels=self.channel_settings[3-i], out_channels=self.channel_settings[2-i], kernel_size=1)
                change = change.cuda()
                up = change(up)
            global_outs.append(feature)

        return global_outs
if __name__ == '__main__':
    x = torch.randn(4, 2048, 12,9 )
    x1 = torch.randn(4, 1024, 24, 18 )
    x2 = torch.randn(4, 512, 48, 36 )
    x3 = torch.randn(4, 256, 96, 72 )
    channel_settings = [2048, 1024, 512, 256]
    model = CANet(channel_settings)
    x = model([x, x1, x2, x3])
    for i in range(2):
        temp = x[i]
        x[i] = x[3-i]
        x[3-i] = temp
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)
    print(x[3].shape)
