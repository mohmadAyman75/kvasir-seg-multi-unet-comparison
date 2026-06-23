from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_channels, hidden_channels, kernel_size=1)
        self.gate_proj = nn.Conv2d(gate_channels, hidden_channels, kernel_size=1)
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=False)
        weights = self.attention(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * weights


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        residual: bool = False,
        attention: bool = False,
    ) -> None:
        super().__init__()
        block = ResidualDoubleConv if residual else DoubleConv
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]

        self.down1 = block(in_channels, channels[0])
        self.down2 = block(channels[0], channels[1])
        self.down3 = block(channels[1], channels[2])
        self.bottleneck = block(channels[2], channels[3])
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.up_block3 = block(channels[3], channels[2])

        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.up_block2 = block(channels[2], channels[1])

        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.up_block1 = block(channels[1], channels[0])

        self.use_attention = attention
        if attention:
            self.att3 = AttentionGate(channels[2], channels[2], channels[1])
            self.att2 = AttentionGate(channels[1], channels[1], channels[0])
            self.att1 = AttentionGate(channels[0], channels[0], max(1, channels[0] // 2))

        self.out = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.down1(x)
        skip2 = self.down2(self.pool(skip1))
        skip3 = self.down3(self.pool(skip2))
        x = self.bottleneck(self.pool(skip3))

        x = self.up3(x)
        if self.use_attention:
            skip3 = self.att3(skip3, x)
        x = torch.cat([skip3, x], dim=1)
        x = self.up_block3(x)

        x = self.up2(x)
        if self.use_attention:
            skip2 = self.att2(skip2, x)
        x = torch.cat([skip2, x], dim=1)
        x = self.up_block2(x)

        x = self.up1(x)
        if self.use_attention:
            skip1 = self.att1(skip1, x)
        x = torch.cat([skip1, x], dim=1)
        x = self.up_block1(x)

        return self.out(x)


class UNetPlusPlus(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.x0_0 = DoubleConv(in_channels, channels[0])
        self.x1_0 = DoubleConv(channels[0], channels[1])
        self.x2_0 = DoubleConv(channels[1], channels[2])
        self.x3_0 = DoubleConv(channels[2], channels[3])

        self.up1_0 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)

        self.up1_1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.up1_2 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)

        self.x0_1 = DoubleConv(channels[0] * 2, channels[0])
        self.x1_1 = DoubleConv(channels[1] * 2, channels[1])
        self.x2_1 = DoubleConv(channels[2] * 2, channels[2])
        self.x0_2 = DoubleConv(channels[0] * 3, channels[0])
        self.x1_2 = DoubleConv(channels[1] * 3, channels[1])
        self.x0_3 = DoubleConv(channels[0] * 4, channels[0])

        self.out = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(self.pool(x0_0))
        x2_0 = self.x2_0(self.pool(x1_0))
        x3_0 = self.x3_0(self.pool(x2_0))

        x0_1 = self.x0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
        x1_1 = self.x1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x2_1 = self.x2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))

        x0_2 = self.x0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], dim=1))
        x1_2 = self.x1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))

        x0_3 = self.x0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))
        return self.out(x0_3)


class TransUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.down1 = DoubleConv(in_channels, channels[0])
        self.down2 = DoubleConv(channels[0], channels[1])
        self.down3 = DoubleConv(channels[1], channels[2])
        self.bottleneck = DoubleConv(channels[2], channels[3])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels[3],
            nhead=transformer_heads,
            dim_feedforward=channels[3] * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.up_block3 = DoubleConv(channels[3], channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.up_block2 = DoubleConv(channels[2], channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.up_block1 = DoubleConv(channels[1], channels[0])
        self.out = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.down1(x)
        skip2 = self.down2(self.pool(skip1))
        skip3 = self.down3(self.pool(skip2))
        x = self.bottleneck(self.pool(skip3))

        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        x = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)

        x = self.up3(x)
        x = self.up_block3(torch.cat([skip3, x], dim=1))
        x = self.up2(x)
        x = self.up_block2(torch.cat([skip2, x], dim=1))
        x = self.up1(x)
        x = self.up_block1(torch.cat([skip1, x], dim=1))
        return self.out(x)


def build_model(name: str, base_channels: int = 32) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNet(base_channels=base_channels)
    if name == "resunet":
        return UNet(base_channels=base_channels, residual=True)
    if name == "attention_unet":
        return UNet(base_channels=base_channels, attention=True)
    if name in {"unet_plus_plus", "unet++"}:
        return UNetPlusPlus(base_channels=base_channels)
    if name == "transunet":
        return TransUNet(base_channels=base_channels)
    raise ValueError(
        "Unknown model "
        f"'{name}'. Choose: unet, resunet, attention_unet, unet_plus_plus, transunet"
    )
