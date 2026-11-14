# HackSpace: Runway Detection

Finding the runway in images taken from an aircraft on approach or landing. This started as a
hackathon entry and has two independent approaches:

- **Classical detector.** Edges plus shape rules. It needs no training data or model weights
  and runs in tens of milliseconds on a 1920x1080 frame on a CPU.
- **U-Net segmentation.** A convolutional network trained on image/mask pairs that predicts a
  runway probability for every pixel.

Both are available from one command-line tool, `runway-detection`, and as a Python package.

## Features

- `classical`: detects runway candidates in files or whole folders, writes a binary mask per
  detection and a JSON report (bounding box, centroid, area, aspect ratio, fill ratio, score).
  Two presets: `improved` (default) and `basic`.
- `split`: turns the dataset's train/test folders into reproducible train/val/test splits.
- `train`: trains a U-Net with augmentation and BCE + Dice loss. Keeps the best checkpoint
  by validation loss and writes `metrics.json` (per-epoch history, and test IoU / precision /
  recall / F1 if you pass a test set) plus `training_curves.png`.
- `predict`: runs a trained checkpoint at full resolution and writes masks, optional red
  overlays, and `predictions.json` with the connected runway regions and coverage.

## Tech stack

Python 3.10+, NumPy, OpenCV, Pillow. The ML part adds PyTorch, Albumentations, Matplotlib and
tqdm. Tests use pytest; linting and formatting use Ruff.

## How it works

### Classical detector (`src/runway_detection/classical.py`)

```
RGB image ─▶ luminance grayscale ─▶ [histogram equalisation] ─▶ gradient magnitude
          ─▶ threshold ─▶ [3x3 dilation] ─▶ 8-connected components ─▶ shape stats
          ─▶ filter / score ─▶ ranked candidates (+ mask of the best one)
```

For each component it measures area, bounding box, aspect ratio (height / width), fill ratio
(area / bounding-box area) and elongation. Components that cover more than half the frame are
dropped as texture clutter.

| Preset     | Gradient          | Equalise | Dilate | Selection                                                    |
|------------|-------------------|----------|--------|--------------------------------------------------------------|
| `improved` | 3x3 Sobel         | yes      | yes    | Weighted score (area, elongation, fill ratio); keeps score > 30 and ranks by score |
| `basic`    | central difference | no      | no     | Hard filters (area 5k-200k px, fill > 0.3, aspect > 1.5 or < 0.7); ranks by area |

Every threshold is a field of `DetectorConfig`, and you can override any of them:
`ClassicalRunwayDetector("improved", edge_threshold=30)`.

This is a heuristic. It assumes the runway is the largest dense, elongated cluster of edges in
the frame, so it struggles when the ground is busy (heavy texture, buildings, clouds). The
U-Net is the approach to use when you have labelled data.

### U-Net (`model.py`, `data.py`, `train.py`, `predict.py`)

- A four-level U-Net (64-128-256-512 channels, 1024 at the bottleneck) with batch norm.
  It outputs logits and trains with `BCEWithLogits + soft Dice`.
- Inputs are resized to 512x512 (configurable) and normalised with ImageNet statistics.
  Training adds flips, 90° rotations, a small affine jitter and brightness/contrast changes.
- Masks are binarised: any non-zero label pixel counts as runway, so colour-coded masks work.
- Optimiser: Adam (lr 1e-4, weight decay 1e-5), with ReduceLROnPlateau on validation loss.
- Metrics are computed over the whole dataset from pixel confusion counts, not averaged per
  batch.
- At inference the probability map is resized back to the input resolution before
  thresholding.
- Checkpoints are plain `state_dict`s and are loaded with `torch.load(..., weights_only=True)`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .                   # classical detector only
pip install -e '.[ml]'             # + U-Net training and inference (PyTorch)
pip install -e '.[ml,dev]'         # + pytest and ruff
```

For a specific CUDA build of PyTorch, install `torch` first using the selector on
[pytorch.org](https://pytorch.org/get-started/locally/), then run the command above.

### Environment variables

None. All configuration goes through command-line flags (see `runway-detection <command> --help`).

## Data

The dataset is not included in this repository, and `.gitignore` excludes `data/`, `archive/`
and model files. The tools expect:

- a folder of RGB images (any of PNG, JPG, BMP, TIFF), and
- for training, a folder of single-channel masks **with the same file names**, where non-zero
  pixels are runway.

The original scripts used this layout, which you can keep:

```
archive/
├── 1920x1080/1920x1080/{train,test}/                # images
└── labels/labels/areas/
    ├── train_labels_1920x1080/                      # masks for train/
    └── test_labels_1920x1080/                       # masks for test/
```

## Usage

```bash
# 1. Classical detection: no model needed
runway-detection classical path/to/images/ --output-dir masks/ --report report.json
runway-detection classical frame.png --preset basic          # JSON report to stdout

# 2. Build a 70/10/20 split: all of train/ stays train; test/ is split 1/3 val, 2/3 test
runway-detection split \
  --train-dir archive/1920x1080/1920x1080/train \
  --test-dir  archive/1920x1080/1920x1080/test \
  --output-dir data/split

# 3. Train the U-Net (val and test masks both come from the original test labels)
runway-detection train \
  --train-images data/split/train --train-masks archive/labels/labels/areas/train_labels_1920x1080 \
  --val-images   data/split/val   --val-masks   archive/labels/labels/areas/test_labels_1920x1080 \
  --test-images  data/split/test  --test-masks  archive/labels/labels/areas/test_labels_1920x1080 \
  --epochs 50 --batch-size 4 --output-dir outputs/

# 4. Predict with the trained model
runway-detection predict data/split/test --weights outputs/best_runway_model.pth \
  --output-dir predictions/ --overlay
```

`python -m runway_detection ...` works the same way. Add `-v` for debug logging.

From Python:

```python
from runway_detection import ClassicalRunwayDetector
from runway_detection.io import load_rgb, save_mask

result = ClassicalRunwayDetector().detect(load_rgb("frame.png"))
if result.success:
    print(result.summary()["main_runway"])
    save_mask(result.mask(), "frame_mask.png")
```

```python
from runway_detection.predict import UNetPredictor
from runway_detection.io import load_rgb

seg = UNetPredictor("outputs/best_runway_model.pth").predict(load_rgb("frame.png"))
print(seg.coverage, seg.regions())
```

## Testing and linting

```bash
pytest                  # 32 tests; the U-Net tests are skipped if PyTorch isn't installed
ruff check .
ruff format --check .
```

The tests use synthetic images, so they need no dataset. They cover:

- the classical pipeline: grayscale weights, edge-map overflow, region measurements, the
  scoring formula, both presets on a synthetic runway, and rejecting flat frames and noise;
- I/O and the CLI end to end;
- the U-Net: output shape for odd input sizes, the losses, the metrics, preprocessing parity
  with Albumentations, dataset pairing and mask binarisation, split determinism, checkpoint
  round-trip, and a short training run on CPU.

## Deployment

This is a command-line tool and library, not a service. To use it elsewhere, install it with
`pip install .` (or build a wheel with `pip wheel . -w dist/`), then copy the
`best_runway_model.pth` checkpoint produced by `train`. The classical detector needs only the
base install.

## Project structure

```
.
├── pyproject.toml              # package metadata, dependencies/extras, pytest + ruff config
├── src/runway_detection/
│   ├── __init__.py             # public API (ClassicalRunwayDetector, DetectorConfig, ...)
│   ├── __main__.py             # python -m runway_detection
│   ├── cli.py                  # classical / split / train / predict subcommands
│   ├── classical.py            # edge + shape detector and presets
│   ├── io.py                   # image/mask loading, saving, directory listing
│   ├── data.py                 # dataset split, image/mask pairing, transforms, Dataset
│   ├── model.py                # U-Net, losses, confusion-count metrics, safe checkpoint loading
│   ├── train.py                # training loop, evaluation, metrics.json, curves
│   └── predict.py              # full-resolution inference, regions, overlays
└── tests/                      # pytest suite on synthetic data
```

## Status

The repository does not include a trained checkpoint or benchmark results. Run `train` on the
dataset to produce your own model and metrics.
