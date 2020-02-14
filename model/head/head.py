from .anchor import anchor
from .anchorless import anchorless


def build_head(cfg):
    if cfg.HEAD == 'anchorless':
        return anchorless.build_anchorless(cfg)
    else:
        return anchor.build_anchor(cfg)

def build_loss(cfg):
    if cfg.HEAD == 'anchorless':
        return anchorless.build_anchorless_loss(cfg)
    else:
        return anchor.build_anchor_loss(cfg)
