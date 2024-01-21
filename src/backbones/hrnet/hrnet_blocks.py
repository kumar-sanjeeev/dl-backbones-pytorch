import logging
from typing import List

import numpy as np
import torch
from torch import nn

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


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
    Basic Block to be used in HrNet.

    It consists of two 3x3 convolutions with batch normalization and ReLU activation.
    First one is used to reduce the number of channels.

    Attributes:
        expansion (int) : Expansion factor for the number of output channels
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
        self.conv1 = conv3x3(inplanes, planes, stride)  # reduce the number of channels
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the BasicBlock.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            torch.tensor : Output tensor.
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


class Bottleneck(nn.Module):
    """
    Bottleneck Block to be used in HrNet.

    It consists of two 1x1 convolutions and one 3x3 convolution with batch normalization and ReLU activation.
    3x3 convolution is used to reduce the number of channels.

    Attributes:
        expansion (int) : Expansion factor for the number of output channels
    """

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        """
        Initializes a Bottleneck.

        Args:
            inplanes (int): Input channel.
            planes (int): Output channel.
            stride (int, optional): Stride of convolution. Default is 1.
            downsample: Downsample function for shortcut connection.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(
            planes, planes, stride=stride
        )  # reduce the number of channels
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the Bottleneck.

        Args:
            x (torch.tensor) : Input tensor.

        Returns:
            torch.tensor : Output tensor.
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


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method,
        multi_scale_output=True,
    ):
        """
        Initializes a HighResolutionModule.

        Args:
            num_branches (int): Number of branches in the module.
            blocks (str): Type of block to be used in the module.
            num_blocks (list): List containing the number of blocks for each branch.
            num_inchannels (list): List containing the number of input channels for each branch.
            num_channels (list): List containing the number of output channels for each branch.
            fuse_method: Method for fusing multi-scale features.
            multi_scale_output (bool, optional): Whether to output multi-scale features. Default is True.
        """
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index])
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                    nn.ReLU(False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


block_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}
