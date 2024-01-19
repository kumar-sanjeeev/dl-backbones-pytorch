import numpy as np
import torch
from torch import nn


def conv3x3(inplanes: int, outplanes: int, stride: int = 1) -> nn.Module:
    """
    3x3 convolution with padding.

    Args:
        inplanes (int): Input channel.
        outplanes (int): Output channel.
        stride (int, optional): Stride of the convolution. Default is 1.

    Returns:
        nn.Module: 3x3 convolution module.
    """
    return nn.Conv2d(
        inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    """
    Basic Block used in ResNet Backbone for ResNet 18, 34.

    Attributes:
        expansion (int): Expansion factor for the number of output channels.
    """

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        """
        Initializes a BasicBlock.

        Args:
            inplanes (int): Input channel.
            planes (int): Output channel.
            stride (int, optional): Stride of convolution. Default is 1.
            downsample: Downsample function for shortcut connection.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BasicBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    """
    Bottleneck Block used in ResNet Backbone for ResNet 50, 101, 152.

    Here, 3x3 convolutions are used for reducing and increasing the dimensions,
    while the bottleneck is a 1x1 convolution. This is opposite to the original paper,
    where 1x1 convolution is used for reducing and increasing the dimensions,
    while the bottleneck is a 3x3 convolution.

    Attributes:
        expansion (int): Expansion factor for the number of output channels.
    """

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        """
        Initializes a BottleneckBlock.

        Args:
            inplanes (int): Input channel.
            planes (int): Output channel.
            stride (int, optional): Stride of convolution. Default is 1.
            downsample: Downsample function for shortcut connection.
        """
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BottleneckBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": BottleneckBlock}
