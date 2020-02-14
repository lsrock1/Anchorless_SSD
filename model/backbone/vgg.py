import torchvision
from torch import nn


# only supports vgg16_bn
def build_vgg():
    vgg = torchvision.models.vgg16_bn(True, True).features

    return nn.ModuleList([
        vgg[:30],
        nn.Sequential(
            vgg[30:-1],
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1), nn.ReLU(inplace=True))
    ])
