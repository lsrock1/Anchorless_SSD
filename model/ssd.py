import torchvision
from torch import nn
import torch
import math
import torch.nn.functional as F

import model.head import head


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(self, in_channels_list, out_channels):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
        """
        super().__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            # inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            inner_top_down = F.interpolate(
                last_inner, size=(int(inner_lateral.shape[-2]), int(inner_lateral.shape[-1])),
                mode='nearest'
            )
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        return tuple(results)


def build_vgg():
    vgg = torchvision.models.vgg16_bn(True, True).features

    return nn.ModuleList([
        vgg[:30],
        nn.Sequential(
            vgg[30:-1],
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1), nn.ReLU(inplace=True))
    ])


def build_extras():
    specs = [
        {'in_channels': 1024, 'out_channels': 512, 'hidden_channels': 256, 'stride': 2},
        {'in_channels': 512, 'out_channels': 256, 'hidden_channels': 128, 'stride': 2},
        {'in_channels': 256, 'out_channels': 256, 'hidden_channels': 128, 'stride': 1},
        {'in_channels': 256, 'out_channels': 256, 'hidden_channels': 128, 'stride': 1}
    ]
    extras = []
    for spec in specs:
        extras.append(
            nn.Sequential(
                nn.Conv2d(spec['in_channels'], spec['hidden_channels'], kernel_size=1), nn.ReLU(True),
                nn.Conv2d(
                    spec['hidden_channels'],
                    spec['out_channels'],
                    kernel_size=3, stride=spec['stride'], padding=spec['stride']-1), nn.ReLU(True)
            ))
    extras = nn.ModuleList(extras)
    return extras


class SSD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.vgg = build_vgg()
        self.extras = build_extras()
        self.fpn = FPN([512, 1024, 512, 256, 256, 256], 256)
        self.head = build_head(cfg)

    def forward(self, x):
        
        results = []
        for layer in self.vgg:
            x = layer(x)
            results.append(x)
        
        for layer in self.extras:
            x = layer(x)
            results.append(x)

        results = self.head(self.fpn(results))

        return results

def build_ssd(cfg):
    return SSD(cfg)