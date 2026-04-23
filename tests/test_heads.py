import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.heads.detection import DetectionHead
from src.heads.segmentation import SegmentationHead
from src.heads.localization import LocalizationHead
from src.heads.depth import DepthHead

B, C = 2, 64   # small channels for fast tests


def fpn_feats(B=B, C=C):
    return [
        torch.randn(B, C, 128, 128),
        torch.randn(B, C, 64,  64),
        torch.randn(B, C, 32,  32),
        torch.randn(B, C, 16,  16),
    ]


# ── Detection ─────────────────────────────────────────────────────────

def test_det_cls_shape():
    head = DetectionHead(in_channels=C, num_classes=10)
    cls, reg, ctr = head(fpn_feats())
    assert cls[0].shape == (B, 10, 128, 128)

def test_det_reg_positive():
    head = DetectionHead(in_channels=C, num_classes=10)
    _, reg, _ = head(fpn_feats())
    for r in reg:
        assert (r > 0).all(), "exp-activated regression must be positive"

def test_det_centerness_shape():
    head = DetectionHead(in_channels=C, num_classes=10)
    _, _, ctr = head(fpn_feats())
    assert ctr[0].shape == (B, 1, 128, 128)


# ── Segmentation ──────────────────────────────────────────────────────

def test_seg_num_classes():
    head = SegmentationHead(in_channels=C, num_classes=4)
    out  = head(fpn_feats())
    assert out.shape[1] == 4

def test_seg_batch_preserved():
    head = SegmentationHead(in_channels=C, num_classes=4)
    out  = head(fpn_feats())
    assert out.shape[0] == B


# ── Localization ──────────────────────────────────────────────────────

def test_loc_location_shape():
    head = LocalizationHead(in_channels=C, num_classes=5)
    out  = head(fpn_feats())
    assert out['location'].shape == (B, 5, 4)

def test_loc_orientation_shape():
    head = LocalizationHead(in_channels=C, num_classes=5)
    out  = head(fpn_feats())
    assert out['orientation'].shape == (B, 5, 2)

def test_loc_keypoints():
    head = LocalizationHead(in_channels=C, num_classes=5, num_keypoints=17)
    out  = head(fpn_feats())
    assert out['keypoints'].shape == (B, 17, 2)


# ── Depth ──────────────────────────────────────────────────────────────

def test_depth_output_shape():
    head = DepthHead(in_channels=C, max_depth=80.0)
    out  = head(fpn_feats())
    assert out.shape[0] == B
    assert out.shape[1] == 1

def test_depth_positive():
    head = DepthHead(in_channels=C, max_depth=80.0)
    out  = head(fpn_feats())
    assert (out >= 0).all()

def test_depth_bounded():
    head = DepthHead(in_channels=C, max_depth=80.0)
    out  = head(fpn_feats())
    assert (out <= 80.0 + 1e-4).all()
