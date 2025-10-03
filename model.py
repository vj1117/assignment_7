import torch
import torch.nn as nn
import torch.nn.functional as F

class Block1(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False):
        super().__init__()
        self.cn1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.ReLU(inplace=True)
        self.cn2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1,
                            padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.cn1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.cn2(x)
        x = self.bn2(x)
        return self.act2(x)

class Block2(nn.Module):
    """Depthwise separable conv: depthwise (k x k) + pointwise 1x1"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False):
        super().__init__()
        self.cn1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.dw = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, groups=64, bias=False)
        self.pw = nn.Conv2d(64, 64, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.cn1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn2(x)
        return self.act2(x)
    
class Block3(nn.Module):
    """Depthwise separable conv: depthwise (k x k) + pointwise 1x1"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False):
        super().__init__()
        self.cn1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.act1 = nn.ReLU(inplace=True)
        self.dw = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, groups=128, bias=False)
        self.pw = nn.Conv2d(128, 128, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU(inplace=True)
           
    def forward(self, x):
        x = self.cn1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn2(x)
        return self.act2(x)

class Block4(nn.Module):
    """Dilation added"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=False):
        super().__init__()
        d3 = 3
        k3 = 3
        padding = d3 * (k3 - 1) // 2

        self.c4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=padding, dilation=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=padding, dilation=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
           
    def forward(self, x):
        x = self.c4(x)
        return x
    
class Model1(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = Block1(3, 32)
        self.c2 = Block2(32, 64)
        self.c3 = Block3(64, 128)
        self.c4 = Block4(128, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x