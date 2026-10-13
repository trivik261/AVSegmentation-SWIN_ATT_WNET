"""
models/attention_unet.py
------------------------
Pure-PyTorch Attention U-Net (and building blocks) for BOTH stages.

- No external deps beyond torch. Ideal on Kaggle (no internet needed).
- Returns (out, ds_out) when deep_supervision=True, else (out, None), so the
  training loop signature is uniform across all backends.
- Configurable in_channels / out_channels / filters / dropout via config.

Used for:
- Stage 1 vessel segmentation (out_channels=2)
- Stage 2 A/V classification (out_channels=4, in_channels=2 overlay)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """Attention U-Net with optional deep supervision.

    forward(x) -> (out, ds_out)
      out:    [B, out_channels, H, W]
      ds_out: [B, out_channels, H, W] if deep_supervision else None
    """

    def __init__(self, in_channels=1, out_channels=2,
                 filters=(32, 64, 128, 256, 512), dropout=0.1,
                 deep_supervision=True):
        super().__init__()
        filters = list(filters)
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = ConvBlock(in_channels, filters[0], dropout)
        self.enc2 = ConvBlock(filters[0], filters[1], dropout)
        self.enc3 = ConvBlock(filters[1], filters[2], dropout)
        self.enc4 = ConvBlock(filters[2], filters[3], dropout)
        self.enc5 = ConvBlock(filters[3], filters[4], dropout)

        self.up5 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.att5 = AttentionBlock(filters[3], filters[3], filters[2])
        self.dec5 = ConvBlock(filters[4], filters[3], dropout)

        self.up4 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.att4 = AttentionBlock(filters[2], filters[2], filters[1])
        self.dec4 = ConvBlock(filters[3], filters[2], dropout)

        self.up3 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.att3 = AttentionBlock(filters[1], filters[1], filters[0])
        self.dec3 = ConvBlock(filters[2], filters[1], dropout)

        self.up2 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.att2 = AttentionBlock(filters[0], filters[0], filters[0] // 2)
        self.dec2 = ConvBlock(filters[1], filters[0], dropout)

        self.out = nn.Conv2d(filters[0], out_channels, 1)
        self.ds_out = nn.Conv2d(filters[2], out_channels, 1) if deep_supervision else None

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d5 = self.up5(e5)
        d5 = torch.cat([self.att5(d5, e4), d5], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat([self.att4(d4, e3), d4], dim=1)
        d4 = self.dec4(d4)

        ds_out = None
        if self.deep_supervision:
            ds_out = self.ds_out(d4)
            ds_out = F.interpolate(ds_out, scale_factor=4, mode="bilinear", align_corners=False)

        d3 = self.up3(d4)
        d3 = torch.cat([self.att3(d3, e2), d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([self.att2(d2, e1), d2], dim=1)
        d2 = self.dec2(d2)

        out = self.out(d2)
        return out, ds_out


def build_pure_pytorch(stage_cfg):
    """Factory: build an AttentionUNet from a stage config dict."""
    arch = stage_cfg.get("arch", "attention_unet")
    if arch != "attention_unet":
        raise ValueError(
            f"pure_pytorch backend only supports 'attention_unet', got '{arch}'"
        )
    return AttentionUNet(
        in_channels=stage_cfg.get("in_channels", 1),
        out_channels=stage_cfg.get("out_channels", 2),
        filters=stage_cfg.get("filters", [32, 64, 128, 256, 512]),
        dropout=stage_cfg.get("dropout", 0.1),
        deep_supervision=stage_cfg.get("deep_supervision", True),
    )
