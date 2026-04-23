import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """
    Semantic segmentation head using all FPN levels merged at the finest scale (P2).
    Two ConvTranspose2d blocks upsample 4× back to full input resolution.
    """

    def __init__(self, in_channels=256, num_classes=19):
        super().__init__()
        self.num_classes = num_classes

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, features):
        target_size = features[0].shape[-2:]
        fused = sum(
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            for f in features
        )
        return self.classifier(self.upsample(self.fuse(fused)))
