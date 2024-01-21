import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore

from src.backbones.hrnet.hrnet import HRNET


@dataclass
class StageConfig:
    num_modules: int
    num_branches: int
    block: str
    num_blocks: List[int]
    num_channels: List[int]
    fuse_method: str


@dataclass
class HRNETConfig:
    stage1: StageConfig
    stage2: StageConfig
    stage3: StageConfig
    stage4: StageConfig
    bn_momentum: float
    dim: Optional[int]
    upsample: bool
    upsample_scale_factor: int
    interpolation_mode: str
    pretrained: bool
    pretrained_path: str


cs = ConfigStore.instance()
cs.store(name="hrnet_upsample_config", node=HRNETConfig)


class HRNETUpsample(nn.Module):
    def __init__(
        self,
        stage1,
        stage2,
        stage3,
        stage4,
        bn_momentum,
        dim,
        upsample,
        upsample_scale_factor,
        interpolation_mode,
        pretrained,
        pretrained_path,
    ):
        super(HRNETUpsample, self).__init__()
        self.stage1_cfg = stage1
        self.stage2_cfg = stage2
        self.stage3_cfg = stage3
        self.stage4_cfg = stage4
        self.bn_momentum = bn_momentum

        self.backbone = HRNET(
            self.stage1_cfg,
            self.stage2_cfg,
            self.stage3_cfg,
            self.stage4_cfg,
            self.bn_momentum,
        )
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained_path)

        if dim is not None:
            self.dim_reduction = nn.Conv2d(
                self.backbone.pre_stage_channels[0],
                dim,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        if upsample:
            self.upsample = nn.Upsample(
                scale_factor=upsample_scale_factor,
                mode=interpolation_mode,
                align_corners=True,
            )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.backbone(x)
        desc_low_res = self.dim_reduction(x[0])
        desc_upsampled = self.upsample(desc_low_res)

        return desc_low_res, desc_upsampled
