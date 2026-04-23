import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CBAM, CrossModalAttention


class ModalityProjector(nn.Module):
    """Projects a secondary modality (depth map / LiDAR BEV) into the FPN feature space."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = max(out_channels // 2, 1)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)


class AttentionFusionModule(nn.Module):
    """
    Multimodal fusion for all four FPN levels.

    Pipeline per level:
      1. Resize secondary projection to this FPN level's spatial resolution.
      2. Cross-modal attention: RGB queries the secondary modality.
      3. Concatenate attended RGB + secondary → 1×1 conv to merge.
      4. CBAM refinement for final attended feature.
    """

    def __init__(self, fpn_channels=256, secondary_in_channels=1, num_heads=8):
        super().__init__()
        self.projector  = ModalityProjector(secondary_in_channels, fpn_channels)
        self.cross_attn = nn.ModuleList([
            CrossModalAttention(fpn_channels, num_heads) for _ in range(4)
        ])
        self.merge = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels * 2, fpn_channels, 1),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(4)
        ])
        self.cbam = nn.ModuleList([CBAM(fpn_channels) for _ in range(4)])

    def forward(self, fpn_features, secondary_input):
        """
        fpn_features    : [P2, P3, P4, P5] each (B, C, H_i, W_i)
        secondary_input : (B, C_sec, H, W)  — depth map or LiDAR projection
        """
        sec_proj = self.projector(secondary_input)
        fused = []
        for i, feat in enumerate(fpn_features):
            sec = F.interpolate(sec_proj, size=feat.shape[-2:],
                                mode='bilinear', align_corners=False)
            attended = self.cross_attn[i](feat, sec)
            merged   = self.merge[i](torch.cat([attended, sec], dim=1))
            fused.append(self.cbam[i](merged))
        return fused
