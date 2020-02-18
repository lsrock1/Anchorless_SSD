from .anchor import anchor
from .anchorless import anchorless


def build_head(cfg, in_channels):
    if cfg.HEAD == 'anchorless':
        return anchorless.build_anchorless(cfg, in_channels)
    else:
        return anchor.build_anchor(cfg, in_channels)
