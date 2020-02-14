from torch import nn
import torch
import math

from .anchorless_inference import make_anchorless_postprocessor


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class AnchorLessHead(nn.Module):
    def __init__(self, cfg, in_channels, out_channel):
        super().__init__()
        assert(all([c == in_channels[0] for c in in_channels]))
        self.tower = []
        self.num_classes = cfg.NUM_CLASSES
        self.postprocessor = make_anchorless_postprocessor(cfg)
        in_channel = in_channels[0]
        for i in range(3):
            if i == 2:
                out_channels = out_channel
            else:
                out_channels = in_channel

            self.tower.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 3, padding=1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True)))
        self.tower = nn.Sequential(*self.tower)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(6)])
        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes, kernel_size=3, stride=1,
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

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def forward(self, x, image_sizes=None):
        logits = []
        bbox_reg = []
        centerness = []

        for l, feature in enumerate(x):
            feature = self.tower(feature)

            logits.append(self.cls_logits(feature))
            centerness.append(self.centerness(feature))

            bbox_pred = self.scales[l](self.bbox_pred(feature))
            bbox_reg.append(torch.exp(bbox_pred))
        
        locations = self.compute_locations(bbox_reg)

        if not self.training:
            return self.postprocessor(locations, box_cls, box_regression, centerness, image_sizes)
        return logits, bbox_reg, centerness, locations
