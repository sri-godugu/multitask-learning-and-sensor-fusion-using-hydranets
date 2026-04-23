import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    """Cross-entropy loss; prediction is bilinearly upsampled to match target resolution."""

    def __init__(self, num_classes=19, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        pred_up = F.interpolate(pred, size=target.shape[-2:],
                                mode='bilinear', align_corners=False)
        return self.ce(pred_up, target.long())
