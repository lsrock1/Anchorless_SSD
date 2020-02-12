import torchvision
import torchvision.ops.nms as nms
from torch import nn
import torch

from collections import namedtuple


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class AnchorLessHead(nn.Module):
    def __init__(self, in_channels, cfg):
        self.tower = []
        self.num_classes = cfg.num_classes
        for i in range(3):
            self.tower.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(True)))
        self.tower = nn.Sequential(*self.tower)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            feature = self.tower(feature)

            logits.append(self.cls_logits(feature))
            centerness.append(self.centerness(feature))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness


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
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, 1)
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

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


def build_vgg():
    vgg = torchvision.models.vgg16_bn(True, True)

    return nn.ModuleList([
        vgg[:23],
        nn.Sequential(
            vgg[23:],
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


def build_ssd(cfg):
    ssd = nn.ModuleList([
        build_vgg(),
        build_extras(),
        FPN([256, 1024, 512, 256, 256, 256], 256),
        AnchorLessHead(256, cfg)
    ])
    return ssd