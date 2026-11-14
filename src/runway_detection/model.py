"""U-Net segmentation model, losses and metrics (requires the ``ml`` extra: PyTorch)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Classic U-Net (Ronneberger et al., 2015) with batch norm.

    ``forward`` returns raw logits; apply ``torch.sigmoid`` to get probabilities.
    Inputs of any size work: decoder features are resized to match each skip connection.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        if not features:
            raise ValueError("features must contain at least one level")
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = in_channels
        for feature in features:
            self.downs.append(_double_conv(channels, feature))
            channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(_double_conv(feature * 2, feature))

        self.bottleneck = _double_conv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        for idx in range(0, len(self.ups), 2):
            skip = skips[-(idx // 2) - 1]
            x = self.ups[idx](x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = self.ups[idx + 1](torch.cat((skip, x), dim=1))

        return self.final_conv(x)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Soft Dice loss computed on sigmoid probabilities."""
    probs = torch.sigmoid(logits).reshape(-1)
    target = target.reshape(-1)
    intersection = (probs * target).sum()
    dice = (2.0 * intersection + smooth) / (probs.sum() + target.sum() + smooth)
    return 1.0 - dice


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy plus Dice loss, the objective used for training."""
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


@dataclass
class ConfusionCounts:
    """Pixel-level confusion counts, accumulated over any number of batches."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.bool()
        target = target.bool()
        self.tp += int((pred & target).sum())
        self.fp += int((pred & ~target).sum())
        self.fn += int((~pred & target).sum())
        self.tn += int((~pred & ~target).sum())

    @staticmethod
    def _ratio(num: int, den: int) -> float:
        return num / den if den else 0.0

    @property
    def iou(self) -> float:
        # Empty prediction on an empty target is a perfect match.
        if self.tp + self.fp + self.fn == 0:
            return 1.0
        return self._ratio(self.tp, self.tp + self.fp + self.fn)

    @property
    def precision(self) -> float:
        return self._ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return self._ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        if self.tp + self.fp + self.fn == 0:
            return 1.0
        return self._ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    def as_dict(self) -> dict[str, float]:
        return {"iou": self.iou, "precision": self.precision, "recall": self.recall, "f1": self.f1}


def load_unet(weights: str, device: torch.device | str = "cpu") -> UNet:
    """Build a :class:`UNet` and load a ``state_dict`` checkpoint safely (``weights_only``)."""
    model = UNet()
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()
