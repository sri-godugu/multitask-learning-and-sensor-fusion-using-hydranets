import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.models.hydranet import HydraNet

BASE = {
    'backbone': 'resnet50',
    'pretrained': False,
    'fpn_channels': 64,
    'backbone_channels': [256, 512, 1024, 2048],
    'num_det_classes': 10,
    'num_seg_classes': 4,
    'num_loc_classes': 5,
    'max_depth': 80.0,
    'use_fusion': False,
    'active_tasks': ['detection', 'segmentation', 'localization', 'depth'],
}


@pytest.fixture(scope='module')
def model():
    return HydraNet(BASE)


def test_all_tasks_present(model):
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert 'detection'   in out
    assert 'segmentation' in out
    assert 'localization' in out
    assert 'depth'        in out


def test_detection_output_keys(model):
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    det = out['detection']
    assert 'cls' in det and 'reg' in det and 'centerness' in det


def test_depth_shape(model):
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    d = out['depth']
    assert d.shape[0] == 1 and d.shape[1] == 1


def test_selective_tasks():
    cfg = dict(BASE, active_tasks=['segmentation', 'depth'])
    m   = HydraNet(cfg)
    x   = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = m(x)
    assert 'segmentation' in out
    assert 'depth'        in out
    assert 'detection'    not in out
    assert 'localization' not in out


def test_fusion_forward():
    cfg = dict(BASE, use_fusion=True, secondary_in_channels=1, fusion_heads=4)
    m   = HydraNet(cfg)
    x   = torch.randn(1, 3, 256, 256)
    sec = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        out = m(x, sec)
    assert 'depth' in out


def test_enable_disable_task():
    cfg = dict(BASE, active_tasks=['segmentation'])
    m   = HydraNet(cfg)
    x   = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = m(x)
    assert 'detection' not in out
    m.enable_task('detection')
    with torch.no_grad():
        out = m(x)
    assert 'detection' in out
    m.disable_task('detection')
    with torch.no_grad():
        out = m(x)
    assert 'detection' not in out
