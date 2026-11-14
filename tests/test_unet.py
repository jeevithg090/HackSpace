"""Tests for the PyTorch parts; skipped when the ``ml`` extra is not installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("albumentations")

from runway_detection.data import (  # noqa: E402
    RunwayDataset,
    build_transforms,
    pair_images_with_masks,
    preprocess,
    split_dataset,
)
from runway_detection.model import (  # noqa: E402
    ConfusionCounts,
    UNet,
    bce_dice_loss,
    dice_loss,
    load_unet,
)
from runway_detection.predict import UNetPredictor, overlay  # noqa: E402

from .conftest import synthetic_runway  # noqa: E402

SMALL = (8, 16)


@pytest.mark.parametrize("size", [(64, 64), (50, 70)])
def test_unet_output_matches_input_size(size: tuple[int, int]) -> None:
    model = UNet(features=SMALL).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, *size))
    assert out.shape == (2, 1, *size)


def test_dice_loss_bounds() -> None:
    target = torch.zeros(1, 1, 4, 4)
    target[..., :2, :] = 1
    perfect = torch.where(target > 0, 20.0, -20.0)
    assert float(dice_loss(perfect, target)) == pytest.approx(0.0, abs=1e-4)
    assert float(dice_loss(-perfect, target)) == pytest.approx(1.0, abs=1e-4)
    assert float(bce_dice_loss(perfect, target)) < float(bce_dice_loss(-perfect, target))


def test_confusion_counts_metrics() -> None:
    pred = torch.tensor([1, 1, 0, 0, 1], dtype=torch.bool)
    target = torch.tensor([1, 0, 1, 0, 1], dtype=torch.bool)
    c = ConfusionCounts()
    c.update(pred, target)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 1)
    assert c.iou == pytest.approx(0.5)
    assert c.precision == pytest.approx(2 / 3)
    assert c.recall == pytest.approx(2 / 3)
    assert c.f1 == pytest.approx(2 / 3)


def test_confusion_counts_empty_is_perfect() -> None:
    c = ConfusionCounts()
    c.update(torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
    assert c.iou == 1.0 and c.f1 == 1.0


def test_preprocess_shape_and_normalisation() -> None:
    img = np.full((30, 40, 3), 255, dtype=np.uint8)
    arr = preprocess(img, 16)
    assert arr.shape == (3, 16, 16) and arr.dtype == np.float32
    assert arr[0, 0, 0] == pytest.approx((1 - 0.485) / 0.229, rel=1e-5)


def test_preprocess_matches_albumentations() -> None:
    image, _ = synthetic_runway(64, 96, box=(30, 5, 20, 50))
    ours = preprocess(image, 32)
    theirs = build_transforms(32, train=False)(image=image)["image"].numpy()
    assert np.allclose(ours, theirs, atol=1e-4)


def _make_split_source(tmp_path: Path) -> tuple[Path, Path, Path]:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for i in range(3):
        img, mask = synthetic_runway(64, 96, box=(30 + i, 5, 20, 50))
        Image.fromarray(img).save(images / f"img{i}.png")
        # Colour-coded labels: any non-zero value is runway.
        Image.fromarray(np.where(mask, 128, 0).astype(np.uint8)).save(masks / f"img{i}.png")
    Image.fromarray(img).save(images / "unlabelled.png")
    return images, masks, tmp_path


def test_dataset_pairs_and_binarises_masks(tmp_path: Path) -> None:
    images, masks, _ = _make_split_source(tmp_path)
    assert [i.name for i, _ in pair_images_with_masks(images, masks)] == [
        "img0.png",
        "img1.png",
        "img2.png",
    ]
    ds = RunwayDataset(images, masks, build_transforms(32, train=True))
    image, mask = ds[0]
    assert image.shape == (3, 32, 32)
    assert mask.shape == (1, 32, 32) and mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
    assert mask.sum() > 0


def test_dataset_without_pairs_raises(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pytest.raises(ValueError):
        RunwayDataset(tmp_path / "a", tmp_path / "b", build_transforms(32, train=False))


def test_split_dataset_is_deterministic(tmp_path: Path) -> None:
    train_dir, test_dir = tmp_path / "train", tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    for i in range(4):
        (train_dir / f"t{i}.png").write_bytes(b"x")
    for i in range(9):
        (test_dir / f"s{i}.png").write_bytes(b"x")

    a = split_dataset(train_dir, test_dir, tmp_path / "out_a", seed=1)
    split_dataset(train_dir, test_dir, tmp_path / "out_b", seed=1)
    assert (a.train, a.val, a.test) == (4, 3, 6)
    names = lambda root, s: sorted(p.name for p in (root / s).iterdir())  # noqa: E731
    assert names(tmp_path / "out_a", "val") == names(tmp_path / "out_b", "val")
    assert set(names(tmp_path / "out_a", "val")).isdisjoint(names(tmp_path / "out_a", "test"))
    with pytest.raises(ValueError):
        split_dataset(train_dir, test_dir, tmp_path / "x", val_fraction=1.5)


def test_checkpoint_roundtrip_and_predictor(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = UNet()
    ckpt = tmp_path / "w.pth"
    torch.save(model.state_dict(), ckpt)
    loaded = load_unet(str(ckpt))
    assert not loaded.training

    predictor = UNetPredictor(loaded, image_size=32, device="cpu")
    image, _ = synthetic_runway(48, 80, box=(30, 5, 15, 40))
    result = predictor.predict(image, threshold=0.5)
    assert result.probability.shape == (48, 80)
    assert result.mask.dtype == bool
    assert 0.0 <= result.probability.min() <= result.probability.max() <= 1.0
    with pytest.raises(ValueError):
        predictor.predict(image[..., 0])


def test_overlay_marks_only_masked_pixels() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.array([[True, False], [False, False]])
    out = overlay(image, mask, alpha=1.0)
    assert out[0, 0].tolist() == [255, 0, 0]
    assert out[1, 1].tolist() == [0, 0, 0]


def test_training_learns_synthetic_runways(tmp_path: Path) -> None:
    from runway_detection.train import TrainConfig, train

    images, masks, _ = _make_split_source(tmp_path)
    cfg = TrainConfig(
        train_images=images,
        train_masks=masks,
        val_images=images,
        val_masks=masks,
        test_images=images,
        test_masks=masks,
        output_dir=tmp_path / "run",
        epochs=2,
        batch_size=2,
        lr=1e-3,
        image_size=32,
        num_workers=0,
        device="cpu",
        plot=True,
    )
    summary = train(cfg)
    assert (tmp_path / "run" / "best_runway_model.pth").exists()
    assert (tmp_path / "run" / "metrics.json").exists()
    assert (tmp_path / "run" / "training_curves.png").exists()
    assert len(summary["history"]) == 2
    assert set(summary["test"]) == {"loss", "iou", "precision", "recall", "f1"}
