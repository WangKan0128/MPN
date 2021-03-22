from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt

class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))

class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""
    def __init__(self, in_channels, reduction_rate=4):
        super(ChannelAttn, self).__init__()
        assert in_channels%reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, int(in_channels/reduction_rate), 1)
        self.conv2 = ConvBlock(int(in_channels/reduction_rate), in_channels, 1)

    def forward(self, x):
        x = F.max_pool2d(x, x.size()[2:])
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x

class SoftAttn(nn.Module):

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.channel_attn = ChannelAttn(in_channels)

    def forward(self, x):
        y_channel = self.channel_attn(x)
        y_channel = torch.sigmoid(y_channel)
        return y_channel
