"""Visual overlays of detections (NumPy and OpenCV only, no PyTorch needed)."""

from __future__ import annotations

import cv2
import numpy as np

HIGHLIGHT = (255, 0, 0)
"""RGB colour used for the mask tint and box outline."""


def overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a red highlight over the masked pixels of an RGB image."""
    out = image_rgb.astype(np.float32).copy()
    red = np.array(HIGHLIGHT, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * red
    return out.astype(np.uint8)


def annotate(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
    label: str | None = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """Tint ``mask`` and optionally draw ``bbox`` = ``(x, y, w, h)`` with a text ``label``."""
    out = np.ascontiguousarray(overlay(image_rgb, mask, alpha))
    if bbox is None:
        return out
    x, y, w, h = bbox
    thickness = max(2, round(min(out.shape[:2]) / 300))
    cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), HIGHLIGHT, thickness)
    if label:
        scale = max(0.5, min(out.shape[:2]) / 900)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y - 6 if y - th - base - 6 >= 0 else y + h + th + 6
        ty = min(ty, out.shape[0] - base - 1)
        tx = min(max(x, 0), max(out.shape[1] - tw - 6, 0))
        cv2.rectangle(out, (tx, ty - th - base), (tx + tw + 6, ty + base), HIGHLIGHT, -1)
        cv2.putText(
            out,
            label,
            (tx + 3, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out
