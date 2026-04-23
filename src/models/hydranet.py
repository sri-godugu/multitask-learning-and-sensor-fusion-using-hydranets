import torch.nn as nn
from ..backbone.resnet import ResNetBackbone
from ..backbone.fpn import FPN
from ..heads.detection import DetectionHead
from ..heads.segmentation import SegmentationHead
from ..heads.localization import LocalizationHead
from ..heads.depth import DepthHead
from ..fusion.fusion_module import AttentionFusionModule


class HydraNet(nn.Module):
    """
    Multi-task HydraNet.

    Architecture:
        Shared ResNet backbone → FPN
        Optional attention-based sensor fusion (RGB + secondary modality)
        Four task heads: Detection · Segmentation · Localization · Depth

    Task heads can be individually enabled/disabled, which is used both at
    inference time (skip unused computation) and during phased fine-tuning.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        fpn_ch = cfg.get('fpn_channels', 256)

        # ── Shared encoder ──────────────────────────────────────────────
        self.backbone = ResNetBackbone(
            name=cfg.get('backbone', 'resnet50'),
            pretrained=cfg.get('pretrained', True),
            freeze_bn=cfg.get('freeze_bn', False),
        )
        self.fpn = FPN(
            in_channels=tuple(cfg.get('backbone_channels', (256, 512, 1024, 2048))),
            out_channels=fpn_ch,
        )

        # ── Sensor fusion (optional) ─────────────────────────────────────
        self.use_fusion = cfg.get('use_fusion', False)
        if self.use_fusion:
            self.fusion = AttentionFusionModule(
                fpn_channels=fpn_ch,
                secondary_in_channels=cfg.get('secondary_in_channels', 1),
                num_heads=cfg.get('fusion_heads', 8),
            )

        # ── Task heads ───────────────────────────────────────────────────
        self.det_head = DetectionHead(
            in_channels=fpn_ch,
            num_classes=cfg.get('num_det_classes', 80),
        )
        self.seg_head = SegmentationHead(
            in_channels=fpn_ch,
            num_classes=cfg.get('num_seg_classes', 19),
        )
        self.loc_head = LocalizationHead(
            in_channels=fpn_ch,
            num_classes=cfg.get('num_loc_classes', 10),
        )
        self.depth_head = DepthHead(
            in_channels=fpn_ch,
            max_depth=cfg.get('max_depth', 100.0),
        )

        self.active_tasks = set(cfg.get(
            'active_tasks', ['detection', 'segmentation', 'localization', 'depth']
        ))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, rgb, secondary=None):
        # Backbone
        c2, c3, c4, c5 = self.backbone(rgb)

        # FPN
        fpn_feats = self.fpn((c2, c3, c4, c5))

        # Sensor fusion
        if self.use_fusion and secondary is not None:
            fpn_feats = self.fusion(fpn_feats, secondary)

        # Task heads
        out = {}
        if 'detection' in self.active_tasks:
            cls_o, reg_o, ctr_o = self.det_head(fpn_feats)
            out['detection'] = {'cls': cls_o, 'reg': reg_o, 'centerness': ctr_o}

        if 'segmentation' in self.active_tasks:
            out['segmentation'] = self.seg_head(fpn_feats)

        if 'localization' in self.active_tasks:
            out['localization'] = self.loc_head(fpn_feats)

        if 'depth' in self.active_tasks:
            out['depth'] = self.depth_head(fpn_feats)

        return out

    # ------------------------------------------------------------------
    # Task control
    # ------------------------------------------------------------------

    def enable_task(self, name):
        self.active_tasks.add(name)

    def disable_task(self, name):
        self.active_tasks.discard(name)
