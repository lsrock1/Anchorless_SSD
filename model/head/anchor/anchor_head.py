import math
import torch
from torch import nn

from .anchor_inference import make_anchor_postprocessor


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __getitem__(self, index):
        return self[index]

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class AnchorBox(nn.Module):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.MIN_DIM
        self.variance = cfg.ANCHOR.VARIANCE
        self.clip = cfg.ANCHOR.CLIP
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

        self.generate_base_anchor(cfg.STRIDES, cfg.MIN_SIZES, cfg.MAX_SIZES, cfg.ANCHOR.RATIOS)

    def generate_base_anchor(self, steps, min_sizes, max_sizes, aspect_ratios):
        base_anchors = [[] for _ in range(len(steps))]
        for idx, (stride, min_size, max_size, aspect_ratio) in enumerate(zip(steps, min_sizes, max_sizes, aspect_ratios)):
            base_anchor = [stride, stride, min_size, min_size]
            base_anchors[idx].append(base_anchor)
            base_anchors[idx].append([
                base_anchor[0],
                base_anchor[1],
                math.sqrt(base_anchor[2] * max_size),
                math.sqrt(base_anchor[3] * max_size)
            ])

            for ratio in aspect_ratio:
                value = math.sqrt(ratio)
                base_anchors[idx].append([
                    base_anchor[0],
                    base_anchor[1],
                    base_anchor[2] * value,
                    base_anchor[3] / value
                ])
                base_anchors[idx].append([
                    base_anchor[0],
                    base_anchor[1],
                    base_anchor[2] / value,
                    base_anchor[3] * value
                ])
        base_anchors = [torch.Tensor(anchor) for anchor in base_anchors]
        self.cell_anchors = BufferList(base_anchors)

    def grid_anchor(self, features, image_size):
        if not torch.is_tensor(image_size):
            image_height, image_width = image_size
            image_size = torch.tensor([image_width, image_height, image_width, image_height])
        # image_height, image_width = image_size
        anchors = []

        for feature, cell_anchor in zip(features, self.cell_anchors):
            height, width = feature.shape[2:]
            grid_y, grid_x = torch.meshgrid(torch.arange(height, device=features[0].device),
                                            torch.arange(width, device=features[0].device))
            grid_x = (grid_x + 0.5).reshape(-1)
            grid_y = (grid_y + 0.5).reshape(-1)

            shifts = torch.stack([grid_x, grid_y, torch.ones_like(grid_x), torch.ones_like(grid_x)], dim=1)
            shifts = (shifts.view(-1, 1, 4) * cell_anchor.view(1, -1, 4)).view(-1, 4)
            shifts = shifts / image_size# torch.tensor([image_width, image_height, image_width, image_height])
            anchors.append(shifts)

        return torch.cat(anchors, dim=0)

    def forward(self, features, image_size):
        output = self.grid_anchor(features, image_size)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class AnchorHead(nn.Module):
    def __init__(self, cfg, out_channels):
        # default anchor 2 + anchor ratio * 2
        num_boxes = [len(ratio) * 2 + 2 for ratio in cfg.ANCHOR.RATIOS]
        self.loc_layers = []
        self.conf_layers = []
        self.num_classes = cfg.NUM_CLASSES

        for out_channel, num_box in zip(out_channels, num_boxes):
            self.loc_layers.append(nn.Conv2d(out_channel, num_boxes * 4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(out_channels, num_boxes * cfg.NUM_CLASSES, kernel_size=3, padding=1))

        self.loc_layers = nn.ModuleList(self.loc_layers)
        self.conf_layers = nn.ModuleList(self.confidences)
        self.anchor_box = AnchorBox(cfg)
        self.postprocessor = make_anchor_postprocessor(cfg)

    def forward(self, x, image_size):
        localizations = []
        confidences = []
        for idx, source in enumerate(x):
            localizations.append(self.loc_layers[idx](source))
            confidences.append(self.conf_layers[idx](source))

        anchors = self.anchor_box(confidences, image_size)

        localizations = [l.permute(0, 2, 3, 1).contiguous() for l in localizations]
        confidences   = [c.permute(0, 2, 3, 1).contiguous() for c in confidences]

        localizations = torch.cat([l.reshape(l.size(0), -1) for l in localizations], 1)
        confidences   = torch.cat([c.reshape(c.size(0), -1) for c in confidences], 1)

        localizations = localizations.reshape(localizations.size(0), -1, 4)
        confidences = confidences.reshape(confidences.size(0), -1, self.num_classes)

        if not self.training:
            return self.postprocessor(localizations, confidences, anchors)
        return localizations, confidences, anchors
