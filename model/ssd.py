import torchvision
from torch import nn
import torch
import math
import torch.nn.functional as F

from model.head.head import build_head
from model.backbone.backbone import build_backbone


class SSD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg, self.backbone.out_channels)

    def forward(self, x, targets=None):
        results = []

        # backbone
        for layer in self.backbone[0]:
            x = layer(x)
            results.append(x)

        # fpn if exists
        if len(self.backbone) > 1:
            results = self.backbone[1](results)

        results = self.head(results, targets)

        return results

def build_ssd(cfg):
    return SSD(cfg)