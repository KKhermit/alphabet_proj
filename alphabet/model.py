from __future__ import annotations

import torch
import torch.nn as nn


class DSConv(nn.Module):
    """Depthwise-separable conv block — same as omr_proj."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyAlphaNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 26, dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            DSConv(16, 24),
            nn.MaxPool2d(2),
            DSConv(24, 32),
            nn.MaxPool2d(2),
            DSConv(32, 48),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class SmallAlphaNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 26, dropout: float = 0.30) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            DSConv(24, 32),
            DSConv(32, 48, stride=2),
            DSConv(48, 64),
            DSConv(64, 96, stride=2),
            DSConv(96, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(cfg: dict) -> nn.Module:
    name = cfg.get("name", "small_cnn").lower()
    in_channels = int(cfg.get("in_channels", 1))
    num_classes = int(cfg.get("num_classes", 26))
    dropout = float(cfg.get("dropout", 0.30))
    if name == "tiny_cnn":
        return TinyAlphaNet(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    if name == "small_cnn":
        return SmallAlphaNet(in_channels=in_channels, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model name: {name!r}. Choose 'tiny_cnn' or 'small_cnn'.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = ["DSConv", "TinyAlphaNet", "SmallAlphaNet", "build_model", "count_parameters"]
