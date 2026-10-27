# FILE: unet_model_new.py
# U-Net with Squeeze-and-Excitation (SE) blocks and Feature-wise Linear Modulation (FiLM)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block: a channel-wise attention mechanism.
    It adaptively re-calibrates channel-wise feature responses.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 + SEBlock"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation Layer.
    Uses a conditioning vector to generate affine transformation parameters (gamma, beta)
    which are then applied to the bottleneck feature map.
    """
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.channels = channels
        self.generator = nn.Sequential(
            nn.Linear(condition_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels * 2)
        )

    def forward(self, x, condition):
        gamma_beta = self.generator(condition)
        gamma = gamma_beta[:, :self.channels].unsqueeze(-1).unsqueeze(-1)
        beta = gamma_beta[:, self.channels:].unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class UNetPotentialField(nn.Module):
    """
    Neural Proposer U-Net that uses FiLM for non-spatial feature fusion (goal conditioning)
    and SE-Blocks for channel-wise attention.
    """
    def __init__(self, n_spatial_channels=4, n_non_spatial_features=2, n_classes_out=1, bilinear_upsample=False, init_channel=64):
        super(UNetPotentialField, self).__init__()

        self.n_in_channels = n_spatial_channels
        self.n_classes_out = n_classes_out
        self.bilinear = bilinear_upsample

        # Encoder path
        self.inc = DoubleConv(self.n_in_channels, init_channel)
        self.down1 = Down(init_channel, init_channel * 2)
        self.down2 = Down(init_channel * 2, init_channel * 4)
        factor = 2 if bilinear_upsample else 1
        self.down3 = Down(init_channel * 4, init_channel * 8 // factor)

        # FiLM Layer for modulating the bottleneck based on non-spatial features
        bottleneck_channels = init_channel * 8 // factor
        self.film_layer = FiLMLayer(bottleneck_channels, n_non_spatial_features)

        # Decoder path
        self.up1 = Up(init_channel * 8, init_channel * 4 // factor, bilinear_upsample)
        self.up2 = Up(init_channel * 4, init_channel * 2 // factor, bilinear_upsample)
        self.up3 = Up(init_channel * 2, init_channel, bilinear_upsample)
        self.outc = OutConv(init_channel, n_classes_out)

    def forward(self, x_spatial: torch.Tensor, x_non_spatial: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Neural Proposer U-Net.

        Args:
            x_spatial (torch.Tensor): Spatial feature maps of shape (B, C_spatial, H, W).
            x_non_spatial (torch.Tensor): Non-spatial conditioning vector of shape (B, C_non_spatial).

        Returns:
            torch.Tensor: Predicted potential field of shape (B, 1, H, W).
        """
        x1 = self.inc(x_spatial)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4_modulated = self.film_layer(x4, x_non_spatial)

        x = self.up1(x4_modulated, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
