from __future__ import annotations

import numpy as np
import pytest


def synthetic_runway(
    height: int = 360, width: int = 640, box: tuple[int, int, int, int] = (280, 60, 80, 260)
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth dark ground with one bright, textured strip at ``box`` = (x, y, w, h).

    The strip is filled with high-contrast speckle (asphalt plus markings), so its edge map
    is dense, which is what the classical detector looks for.
    """
    rng = np.random.default_rng(0)
    xx = np.arange(width, dtype=np.float32)[None, :]
    ground = np.broadcast_to(50 + 20 * xx / width, (height, width)).astype(np.uint8)
    image = np.repeat(ground[..., None], 3, axis=2)
    x, y, w, h = box
    image[y : y + h, x : x + w] = rng.integers(120, 255, size=(h, w, 1), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    mask[y : y + h, x : x + w] = True
    return image, mask


@pytest.fixture
def runway_image() -> tuple[np.ndarray, np.ndarray]:
    return synthetic_runway()
