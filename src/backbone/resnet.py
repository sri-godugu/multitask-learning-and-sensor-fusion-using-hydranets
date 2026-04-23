import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """
    ResNet feature extractor returning C2–C5 feature maps for FPN consumption.
    Strides: C2=4, C3=8, C4=16, C5=32.
    """

    _channels = {
        'resnet50':  (256, 512, 1024, 2048),
        'resnet101': (256, 512, 1024, 2048),
        'resnet34':  (64,  128, 256,  512),
        'resnet18':  (64,  128, 256,  512),
    }

    def __init__(self, name='resnet50', pretrained=True, freeze_bn=False):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        base = getattr(models, name)(weights=weights)

        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # C2
        self.layer2 = base.layer2   # C3
        self.layer3 = base.layer3   # C4
        self.layer4 = base.layer4   # C5

        self.out_channels = self._channels.get(name, (256, 512, 1024, 2048))

        if freeze_bn:
            self._freeze_bn()

    def forward(self, x):
        x  = self.layer0(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
