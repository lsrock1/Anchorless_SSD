from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
from torch import nn


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


class PriorBox(nn.Module):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.variance = cfg['variance'] or [0.1]    
        self.clip = cfg.CLIP

        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
        if pal == 1:
            min_sizes = cfg.ANCHOR_SIZES1
        elif pal == 2:
            min_sizes = cfg.ANCHOR_SIZES2

        self.generate_base_anchor(cfg.STEPS, cfg['min_sizes'], cfg['aspect_ratios'])

    def generate_base_anchor(self, steps, min_sizes, aspect_ratios):
        base_anchors = [[] for _ in range(len(steps))]
        for idx, (stride, size) in enumerate(zip(steps, min_sizes)):
            base_anchor = [stride, stride, size, size]
            
            for ratio in aspect_ratios:
                value = math.sqrt(ratio)
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
            height, width = feature.shape[1:3]
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