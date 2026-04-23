import torch.nn as nn


class DetectionHead(nn.Module):
    """
    FCOS-style anchor-free detection head shared across all FPN levels.
    Outputs per-level: classification logits, bbox regression (exp-activated), centerness.
    """

    def __init__(self, in_channels=256, num_classes=80, num_convs=4):
        super().__init__()
        self.num_classes = num_classes

        def _tower(n):
            layers = []
            for _ in range(n):
                layers += [
                    nn.Conv2d(in_channels, in_channels, 3, padding=1),
                    nn.GroupNorm(min(32, in_channels), in_channels),
                    nn.ReLU(inplace=True),
                ]
            return nn.Sequential(*layers)

        self.cls_tower = _tower(num_convs)
        self.reg_tower = _tower(num_convs)
        self.cls_logits  = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred   = nn.Conv2d(in_channels, 4,          3, padding=1)
        self.centerness  = nn.Conv2d(in_channels, 1,          3, padding=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        cls_outs, reg_outs, ctr_outs = [], [], []
        for feat in features:
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)
            cls_outs.append(self.cls_logits(cls_feat))
            reg_outs.append(self.bbox_pred(reg_feat).exp())
            ctr_outs.append(self.centerness(cls_feat))
        return cls_outs, reg_outs, ctr_outs
