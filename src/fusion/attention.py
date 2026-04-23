import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """SE-style channel attention using both avg- and max-pooled descriptors."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    """CBAM spatial attention using channel-wise avg and max projections."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention → spatial attention."""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial(self.channel(x))


class CrossModalAttention(nn.Module):
    """
    Multi-head cross-modal attention.
    Query from modality A attends to key/value from modality B,
    allowing each modality to pull complementary information from the other.
    Residual connection + GroupNorm stabilises training.
    """

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = max(channels // num_heads, 1)
        inner = self.head_dim * num_heads

        self.q = nn.Conv2d(channels, inner, 1)
        self.k = nn.Conv2d(channels, inner, 1)
        self.v = nn.Conv2d(channels, inner, 1)
        self.out = nn.Conv2d(inner, channels, 1)
        self.norm = nn.GroupNorm(min(32, channels), channels)

    def forward(self, query_feat, key_feat):
        B, C, H, W = query_feat.shape
        N = H * W
        h, d = self.num_heads, self.head_dim

        Q = self.q(query_feat).view(B, h, d, N).permute(0, 1, 3, 2)  # B,h,N,d
        K = self.k(key_feat).view(B, h, d, N).permute(0, 1, 3, 2)
        V = self.v(key_feat).view(B, h, d, N).permute(0, 1, 3, 2)

        attn = (Q @ K.transpose(-2, -1)) * (d ** -0.5)
        attn = attn.softmax(dim=-1)

        out = (attn @ V).permute(0, 1, 3, 2).contiguous().view(B, h * d, H, W)
        return self.norm(query_feat + self.out(out))
