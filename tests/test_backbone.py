import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.backbone.resnet import ResNetBackbone
from src.backbone.fpn import FPN


@pytest.fixture(scope='module')
def backbone():
    return ResNetBackbone('resnet50', pretrained=False)


@pytest.fixture(scope='module')
def fpn():
    return FPN(in_channels=(256, 512, 1024, 2048), out_channels=256)


def test_backbone_feature_shapes(backbone):
    x = torch.randn(2, 3, 512, 512)
    c2, c3, c4, c5 = backbone(x)
    assert c2.shape == (2, 256,  128, 128)
    assert c3.shape == (2, 512,  64,  64)
    assert c4.shape == (2, 1024, 32,  32)
    assert c5.shape == (2, 2048, 16,  16)


def test_fpn_output_count(backbone, fpn):
    x     = torch.randn(2, 3, 512, 512)
    feats = fpn(backbone(x))
    assert len(feats) == 4


def test_fpn_output_channels(backbone, fpn):
    x     = torch.randn(2, 3, 512, 512)
    feats = fpn(backbone(x))
    for f in feats:
        assert f.shape[1] == 256


def test_fpn_spatial_sizes(backbone, fpn):
    x     = torch.randn(2, 3, 512, 512)
    feats = fpn(backbone(x))
    expected = [128, 64, 32, 16]
    for f, s in zip(feats, expected):
        assert f.shape[-1] == s and f.shape[-2] == s


def test_backbone_freeze_bn():
    m = ResNetBackbone('resnet50', pretrained=False, freeze_bn=True)
    import torch.nn as nn
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            for p in mod.parameters():
                assert not p.requires_grad
