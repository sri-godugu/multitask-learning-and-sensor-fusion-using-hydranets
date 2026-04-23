import torch.nn as nn


class LocalizationHead(nn.Module):
    """
    Object localization head predicting per-class center location, size,
    and orientation (sin/cos) from the finest FPN feature via global pooling.
    Optionally predicts keypoints when num_keypoints > 0.
    """

    def __init__(self, in_channels=256, num_classes=10, num_keypoints=0):
        super().__init__()
        self.num_classes   = num_classes
        self.num_keypoints = num_keypoints

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        self.loc_head = nn.Linear(256, num_classes * 4)   # (x, y, w, h) per class
        self.ori_head = nn.Linear(256, num_classes * 2)   # (sin θ, cos θ) per class
        if num_keypoints > 0:
            self.kpt_head = nn.Linear(256, num_keypoints * 2)

    def forward(self, features):
        x = self.pool(features[0]).flatten(1)
        x = self.shared(x)
        out = {
            "location":    self.loc_head(x).view(-1, self.num_classes, 4),
            "orientation": self.ori_head(x).view(-1, self.num_classes, 2),
        }
        if self.num_keypoints > 0:
            out["keypoints"] = self.kpt_head(x).view(-1, self.num_keypoints, 2)
        return out
