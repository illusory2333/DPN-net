from .resnet import *
import torch.nn as nn
import torch
from .globalNet import globalNet
from .refineNet import refineNet

__all__ = ['CPN50', 'CPN101']

class CPN(nn.Module):
    def __init__(self, resnet, output_shape, num_class, pretrained=True):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        self.resnet = resnet
        self.global_net = globalNet(channel_settings, output_shape, num_class)
        self.refine_net = refineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, x):
        res_out = self.resnet(x)
        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)

        return global_outs, refine_out

def CPN50(out_size,num_class,pretrained=True):
    res50 = resnet50(pretrained=pretrained)
    model = CPN(res50, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model

def CPN101(out_size,num_class,pretrained=True):
    res101 = resnet101(pretrained=pretrained)
    model = CPN(res101, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model
# 经过resnet
# torch.Size([4, 2048, 12, 9])
# torch.Size([4, 1024, 24, 18])
# torch.Size([4, 512, 48, 36])
# torch.Size([4, 256, 96, 72])
# 经过globel
# torch.Size([4, 256, 12, 9])
# torch.Size([4, 256, 24, 18])
# torch.Size([4, 256, 48, 36])
# torch.Size([4, 256, 96, 72])
# refine
# torch.Size([4, 17, 96, 72])

# 经过resnet
# torch.Size([4, 2048, 8, 6])
# torch.Size([4, 1024, 16, 12])
# torch.Size([4, 512, 32, 24])
# torch.Size([4, 256, 64, 48])
# 经过globel
# torch.Size([4, 17, 64, 48])
# refine
# torch.Size([4, 17, 64, 48])