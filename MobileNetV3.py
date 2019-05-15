# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class relu(nn.Module):
    def forward(self, x):
        return F.relu(x, inplace=True)


class h_swish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6


class h_sigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6


class se_module(nn.Module):
    def __init__(self, channels, reduction=4):
        super(se_module, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            h_sigmoid()
        )

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.out(y)
        return x * y


class BottleneckV3(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels,
                 kernel_size, stride, non_linearity, se):
        super(BottleneckV3, self).__init__()
        self.stride = stride
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, 1, bias=False),
            nn.BatchNorm2d(exp_channels),
            non_linearity(),
            nn.Conv2d(exp_channels, exp_channels, kernel_size, stride,
                      kernel_size // 2, groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            non_linearity(),
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if se is not None:
            self.out.add_module('se_module', se(out_channels))

        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        y = self.out(x)
        y = y + self.shortcut(x) if self.stride == 1 else y
        return y


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            h_swish(),
            BottleneckV3(16, 16, 16, 3, 1, relu, None),
            BottleneckV3(16, 24, 24, 3, 2, relu, None),
            BottleneckV3(24, 72, 24, 3, 1, relu, None),
            BottleneckV3(24, 72, 40, 5, 2, relu, se_module),
            BottleneckV3(40, 120, 40, 5, 1, relu, se_module),
            BottleneckV3(40, 120, 40, 5, 1, relu, se_module),
            BottleneckV3(40, 240, 80, 3, 2, h_swish, None),
            BottleneckV3(80, 200, 80, 3, 1, h_swish, None),
            BottleneckV3(80, 184, 80, 3, 1, h_swish, None),
            BottleneckV3(80, 184, 80, 3, 1, h_swish, None),
            BottleneckV3(80, 480, 112, 3, 1, h_swish, se_module),
            BottleneckV3(112, 672, 112, 3, 1, h_swish, se_module),
            BottleneckV3(112, 672, 160, 5, 1, h_swish, se_module),
            BottleneckV3(160, 672, 160, 5, 2, h_swish, se_module),
            BottleneckV3(160, 960, 160, 5, 1, h_swish, se_module),
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            h_swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960, 1280, 1),
            h_swish(),
        )
        self.output = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            h_swish(),
            BottleneckV3(16, 16, 16, 3, 2, relu, se_module),
            BottleneckV3(16, 72, 24, 3, 2, relu, None),
            BottleneckV3(24, 88, 24, 3, 1, relu, None),
            BottleneckV3(24, 96, 40, 5, 2, relu, se_module),
            BottleneckV3(40, 240, 40, 5, 1, h_swish, se_module),
            BottleneckV3(40, 240, 40, 5, 1, h_swish, se_module),
            BottleneckV3(40, 120, 48, 5, 1, h_swish, se_module),
            BottleneckV3(48, 144, 48, 5, 1, h_swish, se_module),
            BottleneckV3(48, 288, 96, 5, 2, h_swish, se_module),
            BottleneckV3(96, 576, 96, 5, 1, h_swish, se_module),
            BottleneckV3(96, 576, 96, 5, 1, h_swish, se_module),

            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            h_swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(576, 1280, 1),
            h_swish(),
        )
        self.output = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

