import os
import torch
import torch.nn as nn

from src.backbones.hrnet.hrnet_blocks import block_dict, HighResolutionModule


class HRNET(nn.Module):
    def __init__(self, stage1_cfg, stage2_cfg, stage3_cfg, stage4_cfg, bn_momentum):
        """
        Initializes an HRNET model.

        Args:
            stage1_cfg (dict): Configuration for the first stage.
            stage2_cfg (dict): Configuration for the second stage.
            stage3_cfg (dict): Configuration for the third stage.
            stage4_cfg (dict): Configuration for the fourth stage.
            bn_momentum (float): Momentum for Batch Normalization.
        """
        super(HRNET, self).__init__()
        self.bn_momentum = bn_momentum

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = stage1_cfg
        num_channels = self.stage1_cfg["num_channels"][0]
        block = block_dict[self.stage1_cfg["block"]]
        num_blocks = self.stage1_cfg["num_blocks"][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = stage2_cfg
        num_channels = self.stage2_cfg["num_channels"]
        block = block_dict[self.stage2_cfg["block"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels
        )

        self.stage3_cfg = stage3_cfg
        num_channels = self.stage3_cfg["num_channels"]
        block = block_dict[self.stage3_cfg["block"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels
        )

        self.stage4_cfg = stage4_cfg
        num_channels = self.stage4_cfg["num_channels"]
        block = block_dict[self.stage4_cfg["block"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, self.pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True
        )

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """
        Creates a transition layer between stages.

        Args:
            num_channels_pre_layer (list): Number of channels in the previous layer.
            num_channels_cur_layer (list): Number of channels in the current layer.

        Returns:
            nn.ModuleList: List of transition layers.
        """
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_channels_cur_layer[i], momentum=self.bn_momentum
                            ),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=self.bn_momentum),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """
        Creates a layer with multiple blocks.

        Args:
            block (nn.Module): Type of block to be used.
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            blocks (int): Number of blocks in the layer.
            stride (int, optional): Stride of the layer. Default is 1.

        Returns:
            nn.Sequential: Layer with multiple blocks.
        """
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=self.bn_momentum),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        """
        Creates a stage with multiple HighResolutionModules.

        Args:
            layer_config (dict): Configuration for the stage.
            num_inchannels (list): List of input channels for each branch.
            multi_scale_output (bool, optional): Whether to output multi-scale features. Default is True.

        Returns:
            nn.Sequential: Stage with multiple HighResolutionModules.
            int: Number of output channels from the stage.
        """
        num_modules = layer_config["num_modules"]
        num_branches = layer_config["num_branches"]
        num_blocks = layer_config["num_blocks"]
        num_channels = layer_config["num_channels"]
        block = block_dict[layer_config["block"]]
        fuse_method = layer_config["fuse_method"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used in the last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        """
        Forward pass of the HRNET model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            List[torch.Tensor]: List of output tensors from each branch.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["num_branches"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["num_branches"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["num_branches"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return y_list

    def init_weights(self, pretrained=""):
        """
        Initializes the model weights.

        Args:
            pretrained (str): Path to the pretrained model file.
        """
        print("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            print("=> loading pretrained model {}".format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in model_dict.keys()
            }
            for k, _ in pretrained_dict.items():
                print("=> loading {} pretrained model {}".format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
