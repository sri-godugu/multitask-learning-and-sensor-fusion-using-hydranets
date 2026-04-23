import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.losses.depth_loss import ScaleInvariantDepthLoss
from src.losses.segmentation_loss import SegmentationLoss
from src.losses.localization_loss import LocalizationLoss
from src.losses.multitask_loss import MultitaskLoss


def pos(shape):
    return torch.abs(torch.randn(*shape)) + 0.1


def test_depth_loss_non_negative():
    loss_fn = ScaleInvariantDepthLoss()
    loss    = loss_fn(pos((2, 1, 64, 64)), pos((2, 1, 64, 64)))
    assert loss.item() >= 0


def test_depth_loss_zero_for_identical():
    loss_fn = ScaleInvariantDepthLoss(grad_weight=0.0)
    p = pos((2, 1, 32, 32))
    loss = loss_fn(p, p)
    assert abs(loss.item()) < 1e-4


def test_seg_loss_non_negative():
    loss_fn = SegmentationLoss(num_classes=4)
    pred    = torch.randn(2, 4, 32, 32)
    target  = torch.randint(0, 4, (2, 32, 32))
    assert loss_fn(pred, target).item() >= 0


def test_loc_loss_non_negative():
    loss_fn = LocalizationLoss()
    pred = {'location': torch.randn(2, 5, 4), 'orientation': torch.randn(2, 5, 2)}
    tgt  = {'location': torch.randn(2, 5, 4), 'orientation': torch.randn(2, 5, 2)}
    assert loss_fn(pred, tgt).item() >= 0


def test_multitask_loss_combines():
    mt = MultitaskLoss(
        ['segmentation', 'depth'],
        {'segmentation': SegmentationLoss(4),
         'depth': ScaleInvariantDepthLoss()},
    )
    outputs = {
        'segmentation': torch.randn(2, 4, 32, 32),
        'depth':        pos((2, 1, 32, 32)),
    }
    targets = {
        'segmentation': torch.randint(0, 4, (2, 32, 32)),
        'depth':        pos((2, 1, 32, 32)),
    }
    total, loss_dict = mt(outputs, targets)
    assert 'segmentation' in loss_dict
    assert 'depth'        in loss_dict
    assert total.requires_grad


def test_multitask_log_vars_learnable():
    mt = MultitaskLoss(
        ['depth'],
        {'depth': ScaleInvariantDepthLoss()},
    )
    for name, p in mt.log_vars.items():
        assert p.requires_grad
