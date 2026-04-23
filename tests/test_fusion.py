import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.fusion.attention import ChannelAttention, SpatialAttention, CBAM, CrossModalAttention
from src.fusion.fusion_module import AttentionFusionModule


def test_channel_attention_shape():
    ca  = ChannelAttention(64)
    x   = torch.randn(2, 64, 16, 16)
    out = ca(x)
    assert out.shape == x.shape


def test_spatial_attention_shape():
    sa  = SpatialAttention()
    x   = torch.randn(2, 64, 16, 16)
    out = sa(x)
    assert out.shape == x.shape


def test_cbam_shape():
    cbam = CBAM(64)
    x    = torch.randn(2, 64, 32, 32)
    out  = cbam(x)
    assert out.shape == x.shape


def test_cross_modal_attention_shape():
    attn = CrossModalAttention(64, num_heads=4)
    q    = torch.randn(2, 64, 16, 16)
    k    = torch.randn(2, 64, 16, 16)
    out  = attn(q, k)
    assert out.shape == q.shape


def test_cross_modal_attention_residual():
    """Output should differ from query (attention is not identity)."""
    torch.manual_seed(0)
    attn = CrossModalAttention(64, num_heads=4)
    q    = torch.randn(2, 64, 8, 8)
    k    = torch.randn(2, 64, 8, 8)
    out  = attn(q, k)
    assert not torch.allclose(out, q)


def test_fusion_module_output_shapes():
    fusion = AttentionFusionModule(fpn_channels=64, secondary_in_channels=1, num_heads=4)
    fpn = [
        torch.randn(2, 64, 64, 64),
        torch.randn(2, 64, 32, 32),
        torch.randn(2, 64, 16, 16),
        torch.randn(2, 64,  8,  8),
    ]
    depth  = torch.randn(2, 1, 64, 64)
    fused  = fusion(fpn, depth)
    assert len(fused) == 4
    for f_out, f_in in zip(fused, fpn):
        assert f_out.shape == f_in.shape


def test_fusion_module_multi_channel_secondary():
    fusion = AttentionFusionModule(fpn_channels=64, secondary_in_channels=3, num_heads=4)
    fpn = [torch.randn(1, 64, 32, 32)] * 4
    sec = torch.randn(1, 3, 32, 32)
    out = fusion(fpn, sec)
    assert len(out) == 4
