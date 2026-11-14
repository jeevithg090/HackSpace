"""Inference with a trained U-Net checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import preprocess
from .model import UNet, load_unet


@dataclass
class SegmentationResult:
    probability: np.ndarray
    """Per-pixel runway probability in ``[0, 1]``, at the input image's resolution."""
    mask: np.ndarray
    """Boolean runway mask, ``probability > threshold``."""

    def regions(self, min_area: int = 1000) -> list[dict[str, object]]:
        """Connected runway regions larger than ``min_area`` pixels, largest first."""
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            self.mask.astype(np.uint8), connectivity=8
        )
        found = []
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if area > min_area:
                cx, cy = centroids[label]
                found.append({"bbox": [x, y, w, h], "center": [float(cx), float(cy)], "area": area})
        return sorted(found, key=lambda r: r["area"], reverse=True)  # type: ignore[arg-type,return-value]

    @property
    def coverage(self) -> float:
        return 100.0 * float(self.mask.mean())


class UNetPredictor:
    """Run a trained :class:`UNet` on full-resolution RGB images."""

    def __init__(
        self,
        model: UNet | str | Path,
        image_size: int = 512,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(model, (str, Path)):
            self.model = load_unet(str(model), self.device)
        else:
            self.model = model.to(self.device).eval()
        self.image_size = image_size

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray, threshold: float = 0.5) -> SegmentationResult:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 RGB image, got shape {image_rgb.shape}")
        h, w = image_rgb.shape[:2]
        tensor = torch.from_numpy(preprocess(image_rgb, self.image_size)).unsqueeze(0)
        logits = self.model(tensor.to(self.device))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        prob = np.clip(prob, 0.0, 1.0)
        return SegmentationResult(probability=prob, mask=prob > threshold)


def overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a red highlight over the masked pixels of an RGB image."""
    out = image_rgb.astype(np.float32).copy()
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * red
    return out.astype(np.uint8)
