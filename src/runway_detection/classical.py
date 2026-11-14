"""Classical (non-learned) runway detection.

The pipeline treats a runway as a large, elongated, well-filled cluster of edge pixels:

1. convert to luminance grayscale (optionally histogram-equalised),
2. compute a gradient-magnitude edge map and threshold it,
3. optionally dilate so broken runway markings join up,
4. label 8-connected components,
5. measure each component's shape, drop components that swallow most of the frame
   (textured clutter rather than a runway), and keep the ones that look like a runway,
6. rank the survivors and report the best one as the main runway.

Two presets are provided. ``basic`` uses a central-difference gradient and hard shape
filters, ranking candidates by area. ``improved`` adds contrast equalisation, a Sobel
gradient, dilation and a weighted shape score. Everything is vectorised with NumPy and
OpenCV, so a 1920x1080 frame takes tens of milliseconds on a CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import cv2
import numpy as np

Ranking = Literal["area", "score"]


@dataclass(frozen=True)
class DetectorConfig:
    """Tunable parameters of the classical detector."""

    equalize: bool = True
    gradient: Literal["central", "sobel"] = "sobel"
    edge_threshold: int = 20
    """Threshold applied to the edge map after normalising it to 0-255."""
    dilate: bool = True
    min_component_pixels: int = 300
    min_area: int = 2000
    max_area: int = 300_000
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0
    min_fill_ratio: float = 0.2
    max_frame_fraction: float = 0.5
    """Components covering more than this share of the frame are treated as clutter."""
    ranking: Ranking = "score"
    min_score: float = 30.0
    """Only used with ``ranking="score"``: candidates scoring at or below this are dropped."""


PRESETS: dict[str, DetectorConfig] = {
    "improved": DetectorConfig(),
    "basic": DetectorConfig(
        equalize=False,
        gradient="central",
        edge_threshold=25,
        dilate=False,
        min_component_pixels=500,
        min_area=5000,
        max_area=200_000,
        min_fill_ratio=0.3,
        ranking="area",
    ),
}


@dataclass(frozen=True)
class RegionStats:
    """Shape measurements of one connected component."""

    label: int
    area: int
    bbox: tuple[int, int, int, int]
    """``(x, y, width, height)`` in pixels."""
    center: tuple[float, float]
    """Centroid ``(x, y)``."""
    aspect_ratio: float
    """Bounding-box height divided by width."""
    fill_ratio: float
    """Fraction of the bounding box covered by the component."""
    elongation: float
    """``max(aspect_ratio, 1 / aspect_ratio)``; 1.0 for a square box."""


@dataclass(frozen=True)
class Candidate:
    stats: RegionStats
    score: float


@dataclass
class DetectionResult:
    """Output of :meth:`ClassicalRunwayDetector.detect`."""

    image_shape: tuple[int, ...]
    candidates: list[Candidate]
    labels: np.ndarray = field(repr=False)

    @property
    def success(self) -> bool:
        return bool(self.candidates)

    @property
    def main(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def mask(self, index: int = 0) -> np.ndarray:
        """Boolean mask of the ``index``-th ranked candidate (all-false if there is none)."""
        if index >= len(self.candidates):
            return np.zeros(self.labels.shape, dtype=bool)
        return self.labels == self.candidates[index].stats.label

    def coverage(self) -> float:
        """Percentage of the image covered by the main runway candidate."""
        if self.main is None:
            return 0.0
        h, w = self.image_shape[:2]
        return 100.0 * self.main.stats.area / (h * w)

    def summary(self) -> dict[str, object]:
        """JSON-serialisable summary of the detection."""
        main = self.main
        out: dict[str, object] = {
            "success": self.success,
            "num_candidates": len(self.candidates),
            "image_shape": list(self.image_shape),
        }
        if main is not None:
            s = main.stats
            out["main_runway"] = {
                "score": round(main.score, 2),
                "area": s.area,
                "bbox": list(s.bbox),
                "center": [round(s.center[0], 1), round(s.center[1], 1)],
                "aspect_ratio": round(s.aspect_ratio, 3),
                "fill_ratio": round(s.fill_ratio, 3),
                "elongation": round(s.elongation, 3),
                "coverage_percent": round(self.coverage(), 3),
            }
        return out


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luminance of an RGB image; grayscale input is passed through."""
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    rgb = image[..., :3].astype(np.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def edge_map(gray: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """Binary (0/255) edge map of a grayscale image."""
    src = cv2.equalizeHist(gray) if config.equalize else gray
    src = src.astype(np.float32)
    if config.gradient == "sobel":
        gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
        gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
        magnitude = np.hypot(gx, gy)
    else:
        magnitude = np.zeros_like(src)
        magnitude[1:-1, 1:-1] = np.abs(src[1:-1, 2:] - src[1:-1, :-2]) + np.abs(
            src[2:, 1:-1] - src[:-2, 1:-1]
        )
    # The outermost pixels have no full neighbourhood; ignore them like the gradient does.
    magnitude[[0, -1], :] = 0
    magnitude[:, [0, -1]] = 0

    peak = float(magnitude.max())
    if peak <= 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    normalised = magnitude * (255.0 / peak)
    edges = np.where(normalised > config.edge_threshold, 255, 0).astype(np.uint8)
    if config.dilate:
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    return edges


def measure_regions(binary: np.ndarray, min_pixels: int) -> tuple[np.ndarray, list[RegionStats]]:
    """Label 8-connected components and measure those larger than ``min_pixels``."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    regions: list[RegionStats] = []
    for label in range(1, count):  # label 0 is the background
        x, y, w, h, area = (int(v) for v in stats[label])
        if area <= min_pixels:
            continue
        aspect = h / w
        regions.append(
            RegionStats(
                label=label,
                area=area,
                bbox=(x, y, w, h),
                center=(float(centroids[label][0]), float(centroids[label][1])),
                aspect_ratio=aspect,
                fill_ratio=area / (w * h),
                elongation=max(aspect, 1.0 / aspect),
            )
        )
    return labels, regions


def score_region(stats: RegionStats, config: DetectorConfig) -> float:
    """Weighted runway-likeness score (higher is better, roughly 0-100)."""
    score = 0.0
    if config.min_area < stats.area < config.max_area:
        score += min(stats.area / config.max_area, 1.0) * 30
    if config.min_aspect_ratio < stats.aspect_ratio < config.max_aspect_ratio and (
        stats.aspect_ratio > 2 or stats.aspect_ratio < 0.5
    ):
        score += 25
    if stats.fill_ratio > config.min_fill_ratio:
        score += stats.fill_ratio * 20
    score += stats.fill_ratio * 15
    if stats.elongation > 3:
        score += 10
    return score


def passes_hard_filters(stats: RegionStats, config: DetectorConfig) -> bool:
    """Rule-based filter used by the ``basic`` preset: big, filled and long/narrow."""
    return (
        config.min_area < stats.area < config.max_area
        and stats.fill_ratio > config.min_fill_ratio
        and (stats.aspect_ratio > 1.5 or stats.aspect_ratio < 0.7)
    )


class ClassicalRunwayDetector:
    """Edge- and shape-based runway detector that needs no trained weights."""

    def __init__(self, config: DetectorConfig | str = "improved", **overrides: object) -> None:
        if isinstance(config, str):
            if config not in PRESETS:
                raise ValueError(f"Unknown preset {config!r}; choose from {sorted(PRESETS)}")
            config = PRESETS[config]
        self.config = replace(config, **overrides) if overrides else config

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect runway candidates in an RGB (or grayscale) ``uint8`` image."""
        if image.ndim not in (2, 3) or image.size == 0:
            raise ValueError(f"Expected a non-empty HxW or HxWxC image, got shape {image.shape}")
        cfg = self.config
        edges = edge_map(to_grayscale(image), cfg)
        labels, regions = measure_regions(edges, cfg.min_component_pixels)
        frame_limit = cfg.max_frame_fraction * edges.size
        regions = [r for r in regions if r.area <= frame_limit]

        if cfg.ranking == "area":
            candidates = [
                Candidate(r, score_region(r, cfg)) for r in regions if passes_hard_filters(r, cfg)
            ]
            candidates.sort(key=lambda c: c.stats.area, reverse=True)
        else:
            scored = (Candidate(r, score_region(r, cfg)) for r in regions)
            candidates = [c for c in scored if c.score > cfg.min_score]
            candidates.sort(key=lambda c: c.score, reverse=True)

        return DetectionResult(image_shape=image.shape, candidates=candidates, labels=labels)
