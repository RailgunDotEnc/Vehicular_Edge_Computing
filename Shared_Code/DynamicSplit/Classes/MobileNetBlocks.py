from torch import nn
import torch
from torch import Tensor

class MobileNetV3Block(nn.Module):
    # Convolution Block with Conv2d layer, Batch Normalization and ReLU. Act is an activation function. 
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        kernel_size : int,
        stride : int,
        act = nn.ReLU(),
        groups = 1,
        bn = True,
        bias = False     
        ):
        super().__init__()

        # If k = 1 -> p = 0, k = 3 -> p = 1, k = 5, p = 2. 
        padding = kernel_size // 2 
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.c(x)))
    
class SeBlock(nn.Module):
    # Squeeze and Excitation Block. 
    def __init__(
        self, 
        in_channels : int
        ):
        super().__init__()

        C = in_channels
        r = C // 4
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(C, r, bias=False)
        self.fc2 = nn.Linear(r, C, bias=False)
        self.relu = nn.ReLU() 
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [N, C, H, W].  
        f = self.globpool(x)
        f = torch.flatten(f,1)
        f = self.relu(self.fc1(f))
        f = self.hsigmoid(self.fc2(f))
        f = f[:,:,None,None]
        # f shape: [N, C, 1, 1]  

        scale = x * f
        return scale

# BNeck
class BNeck(nn.Module):
    # MobileNetV3 Block 
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        kernel_size : int, 
        exp_size : int,
        se : bool, 
        act : torch.nn.modules.activation,
        stride : int
        ):
        super().__init__()

        self.add = in_channels == out_channels and stride == 1

        self.block = nn.Sequential(
            MobileNetV3Block(in_channels, exp_size, 1, 1, act),
            MobileNetV3Block(exp_size, exp_size, kernel_size, stride, act, exp_size),
            SeBlock(exp_size) if se == True else nn.Identity(),
            MobileNetV3Block(exp_size, out_channels, 1, 1, act=nn.Identity())
        )

    def forward(self, x: Tensor) -> Tensor:
        res = self.block(x)
        if self.add:
            res = res + x

        return res