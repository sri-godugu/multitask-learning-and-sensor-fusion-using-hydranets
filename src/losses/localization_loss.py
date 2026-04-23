import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizationLoss(nn.Module):
    """Smooth-L1 for location/size + MSE for orientation (sin, cos)."""

    def forward(self, pred, target):
        loc_loss = F.smooth_l1_loss(pred['location'], target['location']) \
            if 'location' in target else torch.tensor(0.0)
        ori_loss = F.mse_loss(pred['orientation'], target['orientation']) \
            if 'orientation' in target else torch.tensor(0.0)
        return loc_loss + ori_loss
