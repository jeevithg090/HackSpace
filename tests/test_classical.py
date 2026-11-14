from __future__ import annotations

import numpy as np
import pytest

from runway_detection.classical import (
    ClassicalRunwayDetector,
    DetectorConfig,
    RegionStats,
    edge_map,
    measure_regions,
    score_region,
    to_grayscale,
)


def test_grayscale_uses_luminance_weights() -> None:
    img = np.zeros((1, 3, 3), dtype=np.uint8)
    img[0, 0] = (255, 0, 0)
    img[0, 1] = (0, 255, 0)
    img[0, 2] = (0, 0, 255)
    assert to_grayscale(img).tolist() == [[76, 150, 29]]


def test_grayscale_passes_through_2d() -> None:
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(to_grayscale(gray), gray)


def test_edge_map_of_flat_image_is_empty() -> None:
    gray = np.full((50, 50), 128, dtype=np.uint8)
    for gradient in ("sobel", "central"):
        cfg = DetectorConfig(gradient=gradient, equalize=False)
        assert not edge_map(gray, cfg).any()


def test_edge_map_does_not_overflow_on_strong_edges() -> None:
    # A 0 -> 255 step makes Sobel magnitudes far above 255; they must not wrap around.
    gray = np.zeros((20, 20), dtype=np.uint8)
    gray[:, 10:] = 255
    edges = edge_map(gray, DetectorConfig(equalize=False, dilate=False))
    assert edges[:, 9:11][1:-1].all()
    assert not edges[:, :8].any() and not edges[:, 12:].any()


def test_measure_regions_reports_bbox_and_shape() -> None:
    binary = np.zeros((100, 100), dtype=np.uint8)
    binary[10:90, 40:50] = 255  # 80 tall, 10 wide
    binary[0:3, 0:3] = 255  # tiny blob, filtered out
    _, regions = measure_regions(binary, min_pixels=50)
    assert len(regions) == 1
    r = regions[0]
    assert r.bbox == (40, 10, 10, 80)
    assert r.area == 800
    assert r.aspect_ratio == pytest.approx(8.0)
    assert r.elongation == pytest.approx(8.0)
    assert r.fill_ratio == pytest.approx(1.0)
    assert r.center == pytest.approx((44.5, 49.5))


def _stats(area: int, w: int, h: int) -> RegionStats:
    aspect = h / w
    return RegionStats(
        label=1,
        area=area,
        bbox=(0, 0, w, h),
        center=(0.0, 0.0),
        aspect_ratio=aspect,
        fill_ratio=area / (w * h),
        elongation=max(aspect, 1 / aspect),
    )


def test_score_prefers_elongated_filled_regions() -> None:
    cfg = DetectorConfig()
    runway_like = _stats(area=40_000, w=100, h=400)  # long, fully filled
    blob = _stats(area=8_000, w=200, h=200)  # square, sparse
    assert score_region(runway_like, cfg) > cfg.min_score > score_region(blob, cfg)


def test_score_components_add_up() -> None:
    cfg = DetectorConfig()
    s = _stats(area=30_000, w=100, h=300)  # aspect 3 (> 2), elongation 3 (not > 3)
    expected = (30_000 / 300_000) * 30 + 25 + 1.0 * 20 + 1.0 * 15
    assert score_region(s, cfg) == pytest.approx(expected)


@pytest.mark.parametrize("preset", ["improved", "basic"])
def test_detects_synthetic_runway(preset: str, runway_image) -> None:
    image, truth = runway_image
    result = ClassicalRunwayDetector(preset).detect(image)
    assert result.success
    mask = result.mask()
    assert mask.shape == truth.shape
    # The detection is edge-based, so it outlines the strip; its box must match the strip.
    x, y, w, h = result.main.stats.bbox
    assert abs(x - 280) <= 3 and abs(y - 60) <= 3
    assert abs(w - 80) <= 6 and abs(h - 260) <= 6
    assert result.summary()["main_runway"]["bbox"] == [x, y, w, h]
    assert 0 < result.coverage() < 100


def test_no_runway_in_flat_image() -> None:
    image = np.full((200, 300, 3), 90, dtype=np.uint8)
    result = ClassicalRunwayDetector().detect(image)
    assert not result.success
    assert result.main is None
    assert not result.mask().any()
    assert result.coverage() == 0.0
    assert result.summary() == {"success": False, "num_candidates": 0, "image_shape": [200, 300, 3]}


def test_unknown_preset_and_bad_shape_raise() -> None:
    with pytest.raises(ValueError, match="Unknown preset"):
        ClassicalRunwayDetector("nope")
    with pytest.raises(ValueError):
        ClassicalRunwayDetector().detect(np.zeros((0, 0), dtype=np.uint8))


def test_overrides_are_applied() -> None:
    det = ClassicalRunwayDetector("basic", edge_threshold=99)
    assert det.config.edge_threshold == 99
    assert det.config.ranking == "area"


def test_frame_filling_clutter_is_not_a_runway() -> None:
    # Pixel noise becomes one giant edge component after equalisation; it must be rejected.
    noise = np.random.default_rng(1).integers(40, 60, size=(360, 640, 3), dtype=np.uint8)
    assert not ClassicalRunwayDetector("improved").detect(noise).success
    lenient = ClassicalRunwayDetector("improved", max_frame_fraction=1.0).detect(noise)
    assert lenient.success and lenient.main.stats.area > 0.5 * 360 * 640
