from .anchorless_head import AnchorLessHead
from .anchorless_loss import AnchorLessLoss


def build_anchorless(cfg, in_channels):
    return AnchorLessHead(cfg, in_channels, cfg.ANCHORLESS.OUT_CHANNELS)
