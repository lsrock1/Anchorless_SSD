from .anchor_head import AnchorHead
from .anchor_loss import AnchorLoss


def build_anchor(cfg, in_channels):
    return AnchorHead(cfg, in_channels)


def build_anchor_loss(cfg):
    return AnchorLoss(cfg)
