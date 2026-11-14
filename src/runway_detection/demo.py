"""Synthetic approach frames for trying the tools without the real dataset.

The frames are generated, not photographs: a sky, smooth terrain and a runway drawn in
perspective (textured asphalt, threshold bars, centreline dashes). Each frame comes with
its ground-truth mask, so the images also work as a tiny dataset for ``split``,
``train`` and ``predict`` smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .io import save_mask


@dataclass(frozen=True)
class DemoScene:
    """Geometry of one synthetic frame, as fractions of the image size."""

    horizon: float
    """Height of the horizon line (0 = top, 1 = bottom)."""
    far_center: float
    """Horizontal position of the runway's far end."""
    near_center: float
    """Horizontal position of the runway's near end (bottom of the frame)."""
    far_width: float
    near_width: float
    far_end: float
    """Height at which the runway starts, between the horizon and the bottom."""
    terrain: tuple[int, int, int]
    """Base RGB colour of the ground."""


SCENES: tuple[DemoScene, ...] = (
    DemoScene(0.30, 0.50, 0.50, 0.016, 0.12, 0.10, (78, 104, 58)),
    DemoScene(0.28, 0.56, 0.46, 0.014, 0.11, 0.05, (112, 98, 70)),
    DemoScene(0.34, 0.45, 0.54, 0.016, 0.12, 0.08, (66, 92, 64)),
    DemoScene(0.31, 0.53, 0.48, 0.014, 0.10, 0.12, (126, 116, 84)),
)


def _smooth_noise(rng: np.random.Generator, height: int, width: int, scale: int) -> np.ndarray:
    """Low-frequency noise in ``[-1, 1]``, used for gentle terrain variation."""
    coarse = rng.uniform(-1.0, 1.0, size=(max(height // scale, 2), max(width // scale, 2)))
    return cv2.resize(coarse.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)


def render_scene(
    scene: DemoScene, width: int = 960, height: int = 540, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Render one frame; returns ``(rgb uint8 image, boolean runway mask)``."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), dtype=np.float32)
    horizon = round(scene.horizon * height)

    # Sky: vertical gradient from pale blue at the horizon to a deeper blue at the top.
    t = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[:, None, None]
    image[:horizon] = (1 - t) * np.array([92, 136, 196]) + t * np.array([176, 202, 226])

    # Terrain: base colour, brighter towards the viewer, plus soft low-frequency patches.
    rows = height - horizon
    depth = np.linspace(0.85, 1.1, rows, dtype=np.float32)[:, None, None]
    patches = _smooth_noise(rng, rows, width, 60)[..., None] * 10
    image[horizon:] = np.array(scene.terrain, dtype=np.float32) * depth + patches

    # Runway trapezoid from its far end down to the bottom edge of the frame.
    top = round(horizon + scene.far_end * (height - horizon) * 0.25)
    half_far, half_near = scene.far_width * width / 2, scene.near_width * width / 2
    cx_far, cx_near = scene.far_center * width, scene.near_center * width
    polygon = np.array(
        [
            (cx_far - half_far, top),
            (cx_far + half_far, top),
            (cx_near + half_near, height - 1),
            (cx_near - half_near, height - 1),
        ],
        dtype=np.int32,
    )
    mask_u8 = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask_u8, [polygon], 1)
    mask = mask_u8.astype(bool)

    asphalt = 70 + rng.normal(0, 28, size=(height, width)).astype(np.float32)
    image[mask] = np.clip(asphalt[mask], 20, 140)[:, None]

    # Markings: interpolate along the runway so they shrink with perspective.
    def point(u: float, v: float) -> tuple[int, int]:
        """``u`` across (-1..1), ``v`` along (0 = far end, 1 = near end)."""
        y = top + v * (height - 1 - top)
        cx = cx_far + v * (cx_near - cx_far)
        half = half_far + v * (half_near - half_far)
        return round(cx + u * half * 0.92), round(y)

    paint = np.zeros((height, width), dtype=np.uint8)
    for v0 in np.arange(0.04, 0.9, 0.12):
        quad = [point(-0.03, v0), point(0.03, v0), point(0.03, v0 + 0.06), point(-0.03, v0 + 0.06)]
        cv2.fillPoly(paint, [np.array(quad, dtype=np.int32)], 1)
    for u0 in np.linspace(-0.85, 0.65, 8):
        quad = [point(u0, 0.0), point(u0 + 0.12, 0.0), point(u0 + 0.12, 0.05), point(u0, 0.05)]
        cv2.fillPoly(paint, [np.array(quad, dtype=np.int32)], 1)
    for side in (-1.0, 1.0):
        cv2.line(paint, point(side * 0.97, 0.0), point(side * 0.97, 1.0), 1, 2)
    image[(paint > 0) & mask] = (232, 232, 226)

    image += rng.normal(0, 1.5, size=image.shape).astype(np.float32)  # sensor noise
    rgb = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return rgb, mask


def write_demo_dataset(
    output_dir: str | Path, count: int = 4, width: int = 960, height: int = 540
) -> list[Path]:
    """Write ``count`` frames to ``output_dir/images`` and masks to ``output_dir/masks``."""
    if count < 1:
        raise ValueError("count must be at least 1")
    root = Path(output_dir)
    (root / "images").mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(count):
        scene = SCENES[i % len(SCENES)]
        rgb, mask = render_scene(scene, width=width, height=height, seed=i)
        name = f"demo_{i + 1:02d}.png"
        path = root / "images" / name
        Image.fromarray(rgb).save(path)
        save_mask(mask, root / "masks" / name)
        written.append(path)
    return written
