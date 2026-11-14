"""Dataset preparation and the PyTorch dataset used for U-Net training."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .io import list_images

if TYPE_CHECKING:
    import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SplitSummary:
    train: int
    val: int
    test: int


def split_dataset(
    train_dir: str | Path,
    test_dir: str | Path,
    output_dir: str | Path,
    val_fraction: float = 1 / 3,
    seed: int = 42,
) -> SplitSummary:
    """Create ``train/val/test`` folders under ``output_dir``.

    All of ``train_dir`` is copied to ``train``; the images in ``test_dir`` are shuffled
    with ``seed`` and divided into ``val`` (``val_fraction``) and ``test`` (the rest).
    With the dataset's own 70/30 train/test split, the default gives roughly 70/10/20.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    train_files = list_images(train_dir)
    test_files = list_images(test_dir)

    shuffled = list(test_files)
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    val_files, held_out = shuffled[:n_val], shuffled[n_val:]

    out = Path(output_dir)
    for name, files in (("train", train_files), ("val", val_files), ("test", held_out)):
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, target / f.name)

    return SplitSummary(train=len(train_files), val=len(val_files), test=len(held_out))


def pair_images_with_masks(image_dir: str | Path, mask_dir: str | Path) -> list[tuple[Path, Path]]:
    """Match images to masks with the same file name; unmatched images are skipped."""
    masks = {p.name: p for p in list_images(mask_dir)}
    return [(img, masks[img.name]) for img in list_images(image_dir) if img.name in masks]


def preprocess(image_rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize and ImageNet-normalise an RGB image into a ``(3, size, size)`` float32 array."""
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    normalised = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalised.transpose(2, 0, 1), dtype=np.float32)


def build_transforms(size: int, train: bool) -> Any:
    """Albumentations pipeline: geometric/photometric augmentation for training only."""
    import albumentations as A  # noqa: N812
    from albumentations.pytorch import ToTensorV2

    steps: list[Any] = [A.Resize(size, size)]
    if train:
        steps += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        ]
    steps += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(steps)


class RunwayDataset:
    """Image/mask pairs for segmentation; any non-zero mask pixel counts as runway.

    Implements the ``torch.utils.data.Dataset`` protocol (``__len__``/``__getitem__``).
    """

    def __init__(self, image_dir: str | Path, mask_dir: str | Path, transform: Any) -> None:
        self.pairs = pair_images_with_masks(image_dir, mask_dir)
        if not self.pairs:
            raise ValueError(f"No image/mask pairs found in {image_dir} and {mask_dir}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise OSError(f"Could not read {image_path} or {mask_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 0).astype(np.uint8)

        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].float().unsqueeze(0)
