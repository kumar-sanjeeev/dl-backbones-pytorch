from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore

import backbones.resnet.resnet as resnet_backbones


@dataclass
class ResNetUpsampleConfig:
    """Configuration with all tunable parameters for the ResNetUpsample network."""
    backbone_type: str
    pretrained: bool = True
    remove_avg_pool_layer: bool = True
    full_conv: bool = True
    upsample: bool = True
    upsample_dim: int = 10


cs = ConfigStore.instance()
cs.store(name="resnet_upsample_config", node=ResNetUpsampleConfig)


class ResNetUpsample(nn.Module):
    """
    ResNet-based network with upsampled output.

    Args:
        backbone_type (str): Type of backbone to use. Valid choices are "resnet18", "resnet34", "resnet50", "resnet101".
        pretrained (bool): Whether to use a pretrained backbone.
        remove_avg_pool_layer (bool): Whether to remove the average pooling layer from the backbone.
        full_conv (bool): Whether to use a fully convolutional layer as the last layer of the backbone.
        upsample (bool): Whether to upsample the output to the input size.
    """

    BACKBONES = {
        "resnet18": resnet_backbones.resnet18,
        "resnet34": resnet_backbones.resnet34,
        "resnet50": resnet_backbones.resnet50,
        "resnet101": resnet_backbones.resnet101,
    }

    def __init__(
        self,
        backbone_type: str,
        pretrained: bool,
        remove_avg_pool_layer: bool,
        full_conv: bool,
        upsample: bool,
        upsample_dim: int,
    ):
        super().__init__()

        if backbone_type not in self.BACKBONES:
            raise ValueError(
                f"Backbone type {backbone_type} is not known. Valid choices are {list(self.BACKBONES.keys())}"
            )

        self.backbone = self.BACKBONES[backbone_type](
            pretrained=pretrained,
            remove_avg_pool_layer=remove_avg_pool_layer,
            full_conv=full_conv,
        )

        # Change the fc layer to a convolutional layer
        self.backbone.fc = nn.Conv2d(
            self.backbone.inplanes, upsample_dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self._normal_initialization(self.backbone.fc)

        if upsample:
            self.upsample = nn.Upsample(
                scale_factor=32, mode="bilinear", align_corners=True
            )

    def _normal_initialization(self, layer: nn.Module) -> None:
        """
        Normal initialization of a layer.

        Args:
            layer (nn.Module): Layer to initialize.
        """
        layer.weight.data.normal_(mean=0.0, std=0.01)
        layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNetUpsample network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Upsampled output tensor.
        """
        x = self.backbone(x)
        upsampled = self.upsample(x)
        return upsampled
