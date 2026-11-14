"""Command-line behaviour: help, validation, error messages, exit codes, demo and overlays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from runway_detection import ClassicalRunwayDetector, __version__
from runway_detection.cli import EXIT_ERROR, EXIT_OK, EXIT_USAGE, build_parser, main
from runway_detection.demo import SCENES, render_scene, write_demo_dataset
from runway_detection.io import load_mask, load_rgb
from runway_detection.viz import annotate


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "quick start" in err
    assert "classical" in err


def test_every_command_has_help_with_examples() -> None:
    parser = build_parser()
    subparsers = next(a for a in parser._actions if a.dest == "command")
    choices = subparsers.choices  # type: ignore[union-attr]
    assert set(choices) == {"demo", "classical", "split", "train", "predict"}
    for name, sub in choices.items():
        assert "example" in sub.format_help(), name


@pytest.mark.parametrize(
    "argv",
    [
        ["predict", "x.png", "--weights", "w.pth", "--threshold", "1.5"],
        ["predict", "x.png", "--weights", "w.pth", "--threshold", "abc"],
        ["demo", "--count", "0"],
        ["demo", "--size", "big"],
        [
            "split",
            "--train-dir",
            "a",
            "--test-dir",
            "b",
            "--output-dir",
            "c",
            "--val-fraction",
            "1",
        ],
        [
            "train",
            "--train-images",
            "a",
            "--train-masks",
            "b",
            "--val-images",
            "c",
            "--val-masks",
            "d",
            "--epochs",
            "-1",
        ],
    ],
)
def test_invalid_values_are_usage_errors(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == EXIT_USAGE


def test_verbose_and_quiet_work_after_the_command(tmp_path: Path) -> None:
    out = tmp_path / "demo"
    assert main(["demo", "-q", "--output-dir", str(out), "--count", "1"]) == EXIT_OK
    assert main(["-v", "demo", "--output-dir", str(out), "--count", "1"]) == EXIT_OK


def test_demo_writes_images_and_masks(tmp_path: Path) -> None:
    out = tmp_path / "demo"
    assert main(["demo", "--output-dir", str(out), "--count", "3", "--size", "320x180"]) == 0
    images = sorted((out / "images").iterdir())
    masks = sorted((out / "masks").iterdir())
    assert (
        [p.name for p in images]
        == [p.name for p in masks]
        == [
            "demo_01.png",
            "demo_02.png",
            "demo_03.png",
        ]
    )
    assert load_rgb(images[0]).shape == (180, 320, 3)
    assert load_mask(masks[0]).any()


def test_demo_scene_is_deterministic_and_detectable() -> None:
    a, mask = render_scene(SCENES[0], seed=0)
    b, _ = render_scene(SCENES[0], seed=0)
    assert np.array_equal(a, b)
    result = ClassicalRunwayDetector().detect(a)
    assert result.success
    found = result.mask()
    iou = (found & mask).sum() / (found | mask).sum()
    assert iou > 0.8


def test_write_demo_dataset_rejects_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_demo_dataset(tmp_path, count=0)


def test_classical_overlay_needs_output_dir(tmp_path: Path) -> None:
    write_demo_dataset(tmp_path, count=1, width=320, height=180)
    assert main(["classical", str(tmp_path / "images"), "--overlay"]) == EXIT_ERROR


def test_classical_writes_overlays(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_demo_dataset(tmp_path, count=1)
    out = tmp_path / "out"
    code = main(["classical", str(tmp_path / "images"), "--output-dir", str(out), "--overlay"])
    assert code == EXIT_OK
    report = json.loads(capsys.readouterr().out)  # JSON goes to stdout, progress to stderr
    assert report[0]["success"] is True
    overlay_path = Path(report[0]["overlay"])
    assert overlay_path.exists()
    assert Image.open(overlay_path).size == (960, 540)


def test_classical_fails_when_nothing_is_readable(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"not an image")
    assert main(["classical", str(tmp_path)]) == EXIT_ERROR


def test_classical_empty_directory_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["classical", str(tmp_path)]) == EXIT_ERROR
    assert "No images found" in capsys.readouterr().err


def test_predict_missing_checkpoint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_demo_dataset(tmp_path, count=1, width=128, height=96)
    code = main(["predict", str(tmp_path / "images"), "--weights", str(tmp_path / "none.pth")])
    assert code == EXIT_ERROR
    assert "Checkpoint not found" in capsys.readouterr().err


def test_predict_corrupt_checkpoint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("torch")
    write_demo_dataset(tmp_path, count=1, width=128, height=96)
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"definitely not a checkpoint")
    code = main(["predict", str(tmp_path / "images"), "--weights", str(bad)])
    assert code == EXIT_ERROR
    assert "Could not load" in capsys.readouterr().err


def test_train_reports_missing_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["train", "--train-images", str(tmp_path / "missing")]
    args += ["--train-masks", str(tmp_path), "--val-images", str(tmp_path)]
    args += ["--val-masks", str(tmp_path), "--output-dir", str(tmp_path / "out")]
    assert main(args) == EXIT_ERROR
    assert "--train-images: directory not found" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_annotate_draws_box_and_label() -> None:
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    mask = np.zeros((120, 200), dtype=bool)
    mask[40:100, 80:120] = True
    out = annotate(image, mask, bbox=(80, 40, 40, 60), label="runway 55")
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert out[40, 100, 0] == 255  # top edge of the box is drawn in red
    assert out[10, 10].sum() == 0  # far from the box the image is untouched
    plain = annotate(image, mask)
    assert plain[70, 100, 0] > 0
    assert plain[0, 0].sum() == 0
