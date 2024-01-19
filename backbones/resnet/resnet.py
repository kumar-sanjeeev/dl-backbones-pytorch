from typing import List

import torch
from torch import nn
from torch.utils import model_zoo

from .resnet_blocks import BasicBlock, BottleneckBlock, blocks_dict

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


class ResNet(nn.Module):
    """
    ResNet Backbone.

    Args:
        block_type (str): Type of block [BASIC, BOTTLENECK].
        layers (List[int]): Number of layers for each block.
        num_classes (int, optional): Number of classes. Default is 1000.
        remove_avg_pool_layer (bool, optional): Remove avg pool layer. Default is False.
    """

    def __init__(
        self,
        block_type: str,
        layers: List[int],
        num_classes: int = 1000,
        remove_avg_pool_layer: bool = False,
    ):
        super(ResNet, self).__init__()
        self.remove_avg_pool_layer = remove_avg_pool_layer
        self.inplanes = 64

        # Resnet Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Resnet Blocks
        self.conv_2x_layer = self._make_layer(block_type, 64, layers[0])
        self.conv_3x_layer = self._make_layer(block_type, 128, layers[1], stride=2)
        self.conv_4x_layer = self._make_layer(block_type, 256, layers[2], stride=2)
        self.conv_5x_layer = self._make_layer(block_type, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * blocks_dict[block_type].expansion, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        conv_2x = self.conv_2x_layer(x)
        conv_3x = self.conv_3x_layer(conv_2x)
        conv_4x = self.conv_4x_layer(conv_3x)
        conv_5x = self.conv_5x_layer(conv_4x)

        if not self.remove_avg_pool_layer:
            x = self.avgpool(conv_5x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block_type: str, planes: int, blocks: int, stride: int = 1):
        """
        Create a layer of blocks.

        Args:
            block_type (str): Type of block (basic or bottleneck).
            planes (int): Output channel.
            blocks (int): Number of blocks.
            stride (int, optional): Stride of the convolution. Default is 1.
        """
        block = blocks_dict[block_type]
        downsample = None

        if self.inplanes != planes * block.expansion or stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample=downsample)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def resnet18(pretrained: bool = False, **kwargs) -> ResNet:
    """
    ResNet 18.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))

    return model


def resnet34(pretrained: bool = False, **kwargs) -> ResNet:
    """
    ResNet 34.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))

    return model


def resnet50(pretrained: bool = False, **kwargs) -> ResNet:
    """
    ResNet 50.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
    """
    model = ResNet(BottleneckBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))

    return model


def resnet101(pretrained: bool = False, **kwargs) -> ResNet:
    """
    ResNet 101.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
    """
    model = ResNet(BottleneckBlock, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))

    return model


def resnet152(pretrained: bool = False, **kwargs) -> ResNet:
    """
    ResNet 152.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
    """
    model = ResNet(BottleneckBlock, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))

    return model
