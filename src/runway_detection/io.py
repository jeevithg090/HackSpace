"""Image loading and saving helpers shared by every detector."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


class ImageLoadError(RuntimeError):
    """Raised when an image cannot be read from disk."""


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as an ``(H, W, 3)`` ``uint8`` RGB array."""
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ImageLoadError(f"Could not read image {path}: {exc}") from exc


def load_mask(path: str | Path) -> np.ndarray:
    """Load a label mask as a boolean ``(H, W)`` array (any non-zero pixel is foreground)."""
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L")) > 0
    except (OSError, ValueError) as exc:
        raise ImageLoadError(f"Could not read mask {path}: {exc}") from exc


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    """Save a boolean or 0/255 mask as an 8-bit grayscale PNG, creating parent folders."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    Image.fromarray(data, mode="L").save(out)
    return out


def list_images(directory: str | Path) -> list[Path]:
    """Return the image files in ``directory`` sorted by name, skipping hidden files."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(
        p
        for p in root.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def iter_inputs(paths: Iterable[str | Path]) -> list[Path]:
    """Expand a mix of files and directories into a flat, sorted list of image paths."""
    images: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            images.extend(list_images(p))
        elif p.is_file():
            images.append(p)
        else:
            raise FileNotFoundError(f"No such file or directory: {p}")
    return images
