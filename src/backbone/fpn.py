import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network.
    Takes (C2, C3, C4, C5) and produces (P2, P3, P4, P5) all with out_channels.
    Top-down pathway fuses coarse semantics into fine spatial features.
    """

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.output = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels
        ])

    def forward(self, features):
        laterals = [l(f) for l, f in zip(self.lateral, features)]

        # Top-down merge
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )

        return [conv(lat) for conv, lat in zip(self.output, laterals)]
