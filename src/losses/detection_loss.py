import torch
import torch.nn as nn
import torch.nn.functional as F


class FCOSLoss(nn.Module):
    """
    Focal loss for FCOS classification + smooth-L1 for regression.
    Both computed per FPN level when ground-truth target lists are provided.
    """

    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, pred, target):
        p = pred.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t) ** self.gamma * ce).mean()

    def forward(self, outputs, targets):
        cls_outs = outputs['cls']
        reg_outs = outputs['reg']
        cls_tgts = targets.get('cls', [])
        reg_tgts = targets.get('reg', [])

        cls_loss = sum(
            self.focal_loss(p, t)
            for p, t in zip(cls_outs, cls_tgts)
            if t is not None
        ) if cls_tgts else torch.tensor(0.0, requires_grad=True)

        reg_loss = sum(
            F.smooth_l1_loss(p, t)
            for p, t in zip(reg_outs, reg_tgts)
            if t is not None
        ) if reg_tgts else torch.tensor(0.0, requires_grad=True)

        return cls_loss + reg_loss
