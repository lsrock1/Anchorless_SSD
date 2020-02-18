from .fpn import FPN
from .vgg import build_vgg

from torch import nn


def build_backbone(cfg):
    assert cfg.BACKBONE_TYPE == 'vgg'
    backbones = []
    if cfg.BACKBONE_TYPE == 'vgg':
        backbones.append(build_vgg())

    if cfg.FPN:
        backbones.append(FPN(backbones[-1].out_channels, cfg.FPN.OUT_CHANNELS))

    out_channels = backbones[-1].out_channels

    backbones = nn.ModuleList(backbones)
    backbones.out_channels = out_channels
    return backbones
