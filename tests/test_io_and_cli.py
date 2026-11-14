from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from runway_detection.cli import main
from runway_detection.io import ImageLoadError, iter_inputs, list_images, load_mask, save_mask

from .conftest import synthetic_runway


def _write(path: Path, array: np.ndarray) -> Path:
    Image.fromarray(array).save(path)
    return path


def test_save_and_load_mask_roundtrip(tmp_path: Path) -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:5, 3:9] = True
    out = save_mask(mask, tmp_path / "nested" / "m.png")
    assert out.exists()
    assert np.array_equal(load_mask(out), mask)
    assert set(np.unique(np.asarray(Image.open(out)))) == {0, 255}


def test_list_images_filters_and_sorts(tmp_path: Path) -> None:
    for name in ("b.png", "a.JPG", ".hidden.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_images(tmp_path)] == ["a.JPG", "b.png"]
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "missing")


def test_iter_inputs_mixes_files_and_dirs(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "x.png").write_bytes(b"")
    single = tmp_path / "y.png"
    single.write_bytes(b"")
    assert [p.name for p in iter_inputs([tmp_path / "d", single])] == ["x.png", "y.png"]
    with pytest.raises(FileNotFoundError):
        iter_inputs([tmp_path / "nope.png"])


def test_cli_classical_writes_masks_and_report(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    runway, _ = synthetic_runway()
    _write(images / "runway.png", runway)
    _write(images / "flat.png", np.full((100, 100, 3), 80, dtype=np.uint8))
    (images / "broken.png").write_bytes(b"not an image")

    out = tmp_path / "out"
    report = tmp_path / "report.json"
    code = main(["classical", str(images), "--output-dir", str(out), "--report", str(report)])
    assert code == 0

    entries = {Path(e["image"]).name: e for e in json.loads(report.read_text())}
    assert entries["runway.png"]["success"] is True
    assert Path(entries["runway.png"]["mask"]).exists()
    assert entries["flat.png"]["success"] is False
    assert "error" in entries["broken.png"]
    assert sorted(p.name for p in out.iterdir()) == ["runway_mask.png"]


def test_cli_reports_missing_input(tmp_path: Path) -> None:
    assert main(["classical", str(tmp_path / "missing")]) == 1


def test_load_rgb_error_type(tmp_path: Path) -> None:
    from runway_detection.io import load_rgb

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"xx")
    with pytest.raises(ImageLoadError):
        load_rgb(bad)
