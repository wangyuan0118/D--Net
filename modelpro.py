# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from functools import partial
import torchvision.ops

class DMSM(nn.Module):
    """动态多尺度激励模块（修复版）"""



class DASC(nn.Module):
    """可变形语义校准器（通道修复版）"""


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample: identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super().__init__()
        width = int(out_channel * (width_per_group / 64)) * groups
        self.conv1 = nn.Conv2d(in_channel, width, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channel*self.expansion, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample: identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, include_top=True,
                 groups=1, width_per_group=64,
                 use_dmsm=False, use_dkam=False, use_dasc=False):
        super().__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], groups=groups, width_per_group=width_per_group)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, groups=groups, width_per_group=width_per_group)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, groups=groups, width_per_group=width_per_group)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, groups=groups, width_per_group=width_per_group)
        self.use_dmsm = use_dmsm
        self.use_dasc = use_dasc
        final_channels = 512 * block.expansion
        if use_dmsm:
            self.dmsm = DMSM(final_channels)
        if use_dasc:
            self.dasc = DASC(in_channels=final_channels, skip_channels=64 * block.expansion)
        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(final_channels, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1, groups=1, width_per_group=64):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, groups=groups, width_per_group=width_per_group))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=groups, width_per_group=width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layer1_out = self.layer1(x)
        x = self.layer2(layer1_out)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.use_dmsm and hasattr(self, 'dmsm'):
            x = self.dmsm(x)
        if self.use_dasc and hasattr(self, 'dasc'):
            x = self.dasc(x, layer1_out)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

def resnet34(num_classes=1000, include_top=True, use_dmsm=False,  use_dasc=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top,
                  use_dmsm=use_dmsm,  use_dasc=use_dasc)

def resnet50(num_classes=1000, include_top=True, use_dmsm=True, use_dasc=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top,
                  use_dmsm=use_dmsm, use_dasc=use_dasc)
