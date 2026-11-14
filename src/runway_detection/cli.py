"""Command-line interface: ``runway-detection {classical,split,train,predict}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .classical import PRESETS, ClassicalRunwayDetector
from .io import ImageLoadError, iter_inputs, load_rgb, save_mask

log = logging.getLogger("runway_detection")


def _cmd_classical(args: argparse.Namespace) -> int:
    detector = ClassicalRunwayDetector(args.preset)
    images = iter_inputs(args.inputs)
    if not images:
        log.error("No images found in %s", ", ".join(args.inputs))
        return 1
    out_dir: Path | None = args.output_dir
    report = []
    detected = 0
    for path in images:
        try:
            result = detector.detect(load_rgb(path))
        except ImageLoadError as exc:
            log.warning("%s", exc)
            report.append({"image": str(path), "error": str(exc)})
            continue
        entry = {"image": str(path), **result.summary()}
        if result.success:
            detected += 1
            if out_dir is not None:
                entry["mask"] = str(save_mask(result.mask(), out_dir / f"{path.stem}_mask.png"))
        report.append(entry)
        log.info("%s: %s", path.name, "runway found" if result.success else "no runway candidate")

    log.info("Detected a runway in %d/%d images", detected, len(images))
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    else:
        print(text)
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    from .data import split_dataset

    summary = split_dataset(args.train_dir, args.test_dir, args.output_dir, args.val_fraction)
    log.info(
        "Split written to %s: train=%d val=%d test=%d",
        args.output_dir,
        summary.train,
        summary.val,
        summary.test,
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, train

    if (args.test_images is None) != (args.test_masks is None):
        log.error("--test-images and --test-masks must be given together")
        return 2
    cfg = TrainConfig(
        train_images=args.train_images,
        train_masks=args.train_masks,
        val_images=args.val_images,
        val_masks=args.val_masks,
        test_images=args.test_images,
        test_masks=args.test_masks,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        plot=not args.no_plot,
    )
    summary = train(cfg)
    log.info(
        "Best validation loss %.4f; checkpoint at %s",
        summary["best_val_loss"],
        summary["checkpoint"],
    )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from PIL import Image

    from .predict import UNetPredictor, overlay

    predictor = UNetPredictor(args.weights, image_size=args.image_size, device=args.device)
    images = iter_inputs(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for path in images:
        try:
            rgb = load_rgb(path)
        except ImageLoadError as exc:
            log.warning("%s", exc)
            continue
        result = predictor.predict(rgb, threshold=args.threshold)
        mask_path = save_mask(result.mask, args.output_dir / f"{path.stem}_mask.png")
        entry: dict[str, object] = {
            "image": str(path),
            "mask": str(mask_path),
            "coverage_percent": round(result.coverage, 3),
            "regions": result.regions(),
        }
        if args.overlay:
            ov_path = args.output_dir / f"{path.stem}_overlay.png"
            Image.fromarray(overlay(rgb, result.mask)).save(ov_path)
            entry["overlay"] = str(ov_path)
        report.append(entry)
        log.info("%s: %.2f%% runway pixels", path.name, result.coverage)
    (args.output_dir / "predictions.json").write_text(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runway-detection", description="Detect runways in approach/landing images."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("classical", help="edge/shape-based detector (no model needed)")
    p.add_argument("inputs", nargs="+", help="image files and/or directories")
    p.add_argument("--preset", choices=sorted(PRESETS), default="improved")
    p.add_argument("--output-dir", type=Path, help="write a mask PNG per detection here")
    p.add_argument("--report", type=Path, help="write the JSON report here instead of stdout")
    p.set_defaults(func=_cmd_classical)

    p = sub.add_parser("split", help="create train/val/test folders from the dataset")
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--val-fraction",
        type=float,
        default=1 / 3,
        help="share of the original test set that becomes validation",
    )
    p.set_defaults(func=_cmd_split)

    p = sub.add_parser("train", help="train the U-Net segmentation model")
    p.add_argument("--train-images", type=Path, required=True)
    p.add_argument("--train-masks", type=Path, required=True)
    p.add_argument("--val-images", type=Path, required=True)
    p.add_argument("--val-masks", type=Path, required=True)
    p.add_argument("--test-images", type=Path)
    p.add_argument("--test-masks", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", help="e.g. cpu, cuda, cuda:1 (default: auto)")
    p.add_argument("--no-plot", action="store_true", help="skip training_curves.png")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("predict", help="segment images with a trained U-Net")
    p.add_argument("inputs", nargs="+", help="image files and/or directories")
    p.add_argument("--weights", type=Path, required=True, help="checkpoint from `train`")
    p.add_argument("--output-dir", type=Path, default=Path("predictions"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--device")
    p.add_argument("--overlay", action="store_true", help="also save red overlay images")
    p.set_defaults(func=_cmd_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    except ModuleNotFoundError as exc:
        log.error("%s. Install the ML extras with: pip install -e '.[ml]'", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
