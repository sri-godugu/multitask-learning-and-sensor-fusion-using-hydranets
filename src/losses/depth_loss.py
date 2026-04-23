import torch
import torch.nn as nn


class ScaleInvariantDepthLoss(nn.Module):
    """
    Scale-invariant log-depth loss (Eigen et al., 2014) with an added
    gradient-matching term that penalises blurry depth boundaries.
    """

    def __init__(self, alpha=0.5, grad_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.grad_weight = grad_weight

    def forward(self, pred, target):
        valid = (target > 0) & (pred > 0)
        if valid.sum() == 0:
            return pred.sum() * 0.0

        log_diff = torch.log(pred[valid] + 1e-7) - torch.log(target[valid] + 1e-7)
        si_loss  = (log_diff ** 2).mean() - self.alpha * (log_diff.mean() ** 2)

        # Gradient loss — penalise depth boundary errors
        pred_dx  = pred[:, :, :, 1:]  - pred[:, :, :, :-1]
        pred_dy  = pred[:, :, 1:, :]  - pred[:, :, :-1, :]
        tgt_dx   = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgt_dy   = target[:, :, 1:, :]  - target[:, :, :-1, :]
        grad_loss = ((pred_dx - tgt_dx).abs() + (pred_dy - tgt_dy).abs()).mean()

        return si_loss + self.grad_weight * grad_loss
