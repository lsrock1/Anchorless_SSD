import torchvision
from torch import nn


# only supports vgg16_bn
def build_vgg():
    vgg = torchvision.models.vgg16_bn(True, True).features
    vgg_with_extras = [
        vgg[:30],
        nn.Sequential(
            vgg[30:-1],
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1), nn.ReLU(inplace=True))
    ]

    extras_spec = [
        {'in_channels': 1024, 'out_channels': 512, 'hidden_channels': 256, 'stride': 2},
        {'in_channels': 512, 'out_channels': 256, 'hidden_channels': 128, 'stride': 2},
        {'in_channels': 256, 'out_channels': 256, 'hidden_channels': 128, 'stride': 1},
        {'in_channels': 256, 'out_channels': 256, 'hidden_channels': 128, 'stride': 1}
    ]

    for spec in extras_spec:
        vgg_with_extras.append(
            nn.Sequential(
                nn.Conv2d(spec['in_channels'], spec['hidden_channels'], kernel_size=1), nn.ReLU(True),
                nn.Conv2d(
                    spec['hidden_channels'],
                    spec['out_channels'],
                    kernel_size=3, stride=spec['stride'], padding=spec['stride']-1), nn.ReLU(True)
            ))

    vgg_with_extras = nn.ModuleList(vgg_with_extras)
    vgg_with_extras.out_channels = [512, 1024, 512, 256, 256, 256]
    return vgg_with_extras
