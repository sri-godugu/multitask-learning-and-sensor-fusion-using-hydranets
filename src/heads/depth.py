import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    """
    Monocular depth estimation head using all four FPN skip connections.
    Decoder progressively merges coarse (P5) and fine (P2) features,
    producing a per-pixel depth map at half the input resolution (2× upsampled at end).
    Sigmoid output is scaled to [0, max_depth].
    """

    def __init__(self, in_channels=256, max_depth=100.0):
        super().__init__()
        self.max_depth = max_depth

        self.upconv4 = self._block(in_channels,              128)
        self.upconv3 = self._block(128 + in_channels,        64)
        self.upconv2 = self._block(64  + in_channels,        32)
        self.upconv1 = self._block(32  + in_channels,        16)

        self.depth_out = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _block(in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ELU(inplace=True),
        )

    def forward(self, features):
        p2, p3, p4, p5 = features

        x = self.upconv4(p5)
        x = F.interpolate(x, size=p4.shape[-2:], mode='bilinear', align_corners=False)
        x = self.upconv3(torch.cat([x, p4], dim=1))
        x = F.interpolate(x, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        x = self.upconv2(torch.cat([x, p3], dim=1))
        x = F.interpolate(x, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.upconv1(torch.cat([x, p2], dim=1))

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.depth_out(x) * self.max_depth
