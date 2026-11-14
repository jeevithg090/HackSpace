"""Command-line interface: ``runway-detection {demo,classical,split,train,predict}``.

Exit codes: 0 success, 1 runtime error (missing files, unreadable inputs, missing extras),
2 invalid command-line usage, 130 interrupted with Ctrl+C.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .classical import PRESETS, ClassicalRunwayDetector
from .io import ImageLoadError, iter_inputs, load_rgb, save_mask

log = logging.getLogger("runway_detection")

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130
ML_HINT = "Install the ML extras with: pip install -e '.[ml]'"


class CliError(Exception):
    """An expected failure with a message meant for the user (no traceback)."""


class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keep hand-written line breaks and append meaningful defaults to option help."""

    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=32, width=min(_terminal_width(), 100))

    def _get_help_string(self, action: argparse.Action) -> str | None:
        text = action.help or ""
        default = action.default
        if (
            default is None
            or default is False
            or default is argparse.SUPPRESS
            or isinstance(action, argparse._HelpAction | argparse._VersionAction)
            or "%(default)" in text
        ):
            return text
        if isinstance(default, tuple):  # --size
            shown = "x".join(str(v) for v in default)
        elif isinstance(default, float):
            shown = f"{default:.4g}"
        else:
            shown = str(default)
        return f"{text} (default: {shown})" if text else f"default: {shown}"


def _terminal_width() -> int:
    import shutil

    return shutil.get_terminal_size((100, 24)).columns


class _LogFormatter(logging.Formatter):
    """Plain messages for progress; ``warning:`` / ``error:`` prefixes for problems."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"error: {message}"
        if record.levelno >= logging.WARNING:
            return f"warning: {message}"
        if record.levelno <= logging.DEBUG:
            return f"debug: {message}"
        return message


# --------------------------------------------------------------------------- validators


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def _open_unit_interval(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError(f"must be strictly between 0 and 1, got {value}")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return value


def _size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected WIDTHxHEIGHT, e.g. 960x540, got {text!r}"
        ) from None
    if width < 64 or height < 64:
        raise argparse.ArgumentTypeError("width and height must both be at least 64 pixels")
    return width, height


# --------------------------------------------------------------------------- helpers


def _collect_images(inputs: Sequence[str]) -> list[Path]:
    images = iter_inputs(inputs)
    if not images:
        raise CliError(
            f"No images found in {', '.join(inputs)} "
            "(looked for .png, .jpg, .jpeg, .bmp, .tif and .tiff files)"
        )
    return images


def _progress(index: int, total: int) -> str:
    width = len(str(total))
    return f"[{index:>{width}}/{total}]"


# --------------------------------------------------------------------------- commands


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import write_demo_dataset

    width, height = args.size
    paths = write_demo_dataset(args.output_dir, count=args.count, width=width, height=height)
    out = args.output_dir
    log.info(
        "Wrote %d synthetic frames to %s and their masks to %s",
        len(paths),
        out / "images",
        out / "masks",
    )
    log.info(
        "Try it: runway-detection classical %s --output-dir %s --overlay",
        out / "images",
        out / "detections",
    )
    return EXIT_OK


def _cmd_classical(args: argparse.Namespace) -> int:
    from PIL import Image

    from .viz import annotate

    if args.overlay and args.output_dir is None:
        raise CliError("--overlay needs --output-dir to know where to write the images")
    detector = ClassicalRunwayDetector(args.preset)
    images = _collect_images(args.inputs)
    out_dir: Path | None = args.output_dir
    report = []
    detected = failed = 0
    for index, path in enumerate(images, start=1):
        step = _progress(index, len(images))
        try:
            rgb = load_rgb(path)
        except ImageLoadError as exc:
            failed += 1
            log.warning("%s %s", step, exc)
            report.append({"image": str(path), "error": str(exc)})
            continue
        result = detector.detect(rgb)
        entry = {"image": str(path), **result.summary()}
        if result.success and result.main is not None:
            detected += 1
            main = result.main
            if out_dir is not None:
                entry["mask"] = str(save_mask(result.mask(), out_dir / f"{path.stem}_mask.png"))
            if args.overlay and out_dir is not None:
                label = f"runway {main.score:.0f}"
                annotated = annotate(rgb, result.mask(), main.stats.bbox, label)
                ov_path = out_dir / f"{path.stem}_overlay.png"
                Image.fromarray(annotated).save(ov_path)
                entry["overlay"] = str(ov_path)
            log.info(
                "%s %s: runway found (score %.1f, %.1f%% of frame, %d candidate%s)",
                step,
                path.name,
                main.score,
                result.coverage(),
                len(result.candidates),
                "" if len(result.candidates) == 1 else "s",
            )
        else:
            log.info("%s %s: no runway candidate", step, path.name)
        report.append(entry)

    readable = len(images) - failed
    log.info(
        "Done: runway found in %d of %d image%s%s",
        detected,
        readable,
        "" if readable == 1 else "s",
        f" ({failed} could not be read)" if failed else "",
    )
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
        log.info("Report written to %s", args.report)
    else:
        print(text)
    if out_dir is not None and detected:
        log.info("Masks%s written to %s", " and overlays" if args.overlay else "", out_dir)
    return EXIT_ERROR if readable == 0 else EXIT_OK


def _cmd_split(args: argparse.Namespace) -> int:
    from .data import split_dataset

    summary = split_dataset(
        args.train_dir, args.test_dir, args.output_dir, args.val_fraction, seed=args.seed
    )
    log.info(
        "Split written to %s: train=%d val=%d test=%d",
        args.output_dir,
        summary.train,
        summary.val,
        summary.test,
    )
    return EXIT_OK


def _cmd_train(args: argparse.Namespace) -> int:
    if (args.test_images is None) != (args.test_masks is None):
        raise CliError("--test-images and --test-masks must be given together")
    for flag in (
        "train_images",
        "train_masks",
        "val_images",
        "val_masks",
        "test_images",
        "test_masks",
    ):
        folder = getattr(args, flag)
        if folder is not None and not folder.is_dir():
            raise CliError(f"--{flag.replace('_', '-')}: directory not found: {folder}")

    from .train import TrainConfig, train

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
    log.info("Metrics written to %s", Path(cfg.output_dir) / "metrics.json")
    return EXIT_OK


def _cmd_predict(args: argparse.Namespace) -> int:
    weights: Path = args.weights
    if not weights.is_file():
        raise CliError(
            f"Checkpoint not found: {weights} (create one with `runway-detection train`)"
        )
    images = _collect_images(args.inputs)

    from PIL import Image

    from .predict import UNetPredictor
    from .viz import overlay

    try:
        predictor = UNetPredictor(weights, image_size=args.image_size, device=args.device)
    except ModuleNotFoundError:
        raise
    except Exception as exc:  # torch raises many types for corrupt or foreign files
        reason = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        detail = f"{type(exc).__name__}: {reason}" if reason else type(exc).__name__
        raise CliError(
            f"Could not load {weights} as a U-Net checkpoint saved by `train` ({detail[:200]})"
        ) from exc
    log.info("Loaded %s on %s", weights, predictor.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    failed = 0
    for index, path in enumerate(images, start=1):
        step = _progress(index, len(images))
        try:
            rgb = load_rgb(path)
        except ImageLoadError as exc:
            failed += 1
            log.warning("%s %s", step, exc)
            continue
        result = predictor.predict(rgb, threshold=args.threshold)
        mask_path = save_mask(result.mask, args.output_dir / f"{path.stem}_mask.png")
        regions = result.regions()
        entry: dict[str, object] = {
            "image": str(path),
            "mask": str(mask_path),
            "coverage_percent": round(result.coverage, 3),
            "regions": regions,
        }
        if args.overlay:
            ov_path = args.output_dir / f"{path.stem}_overlay.png"
            Image.fromarray(overlay(rgb, result.mask)).save(ov_path)
            entry["overlay"] = str(ov_path)
        report.append(entry)
        log.info(
            "%s %s: %.2f%% runway pixels, %d region%s",
            step,
            path.name,
            result.coverage,
            len(regions),
            "" if len(regions) == 1 else "s",
        )
    report_path = args.output_dir / "predictions.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info(
        "Done: %d image%s segmented%s; results in %s",
        len(report),
        "" if len(report) == 1 else "s",
        f" ({failed} could not be read)" if failed else "",
        args.output_dir,
    )
    return EXIT_ERROR if not report else EXIT_OK


# --------------------------------------------------------------------------- parser

MAIN_EPILOG = """\
quick start (no dataset needed):
  runway-detection demo                                   # write synthetic frames to ./demo
  runway-detection classical demo/images --output-dir demo/detections --overlay

with the real dataset:
  runway-detection split --train-dir ... --test-dir ... --output-dir data/split
  runway-detection train --train-images ... --train-masks ... --val-images ... --val-masks ...
  runway-detection predict data/split/test --weights outputs/best_runway_model.pth --overlay

Run `runway-detection <command> --help` for every option of a command.
"""


def _add_common(parser: argparse.ArgumentParser, *, top_level: bool) -> None:
    """``-v``/``-q`` work both before and after the command name."""
    default = False if top_level else argparse.SUPPRESS
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v", "--verbose", action="store_true", default=default, help="show debug messages"
    )
    group.add_argument(
        "-q", "--quiet", action="store_true", default=default, help="only show warnings and errors"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runway-detection",
        description=(
            "Find the runway in images taken from an aircraft on approach or landing.\n"
            "Two approaches: a classical edge/shape detector (no training needed) and a\n"
            "U-Net segmentation model (needs the `ml` extra and labelled data)."
        ),
        epilog=MAIN_EPILOG,
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _add_common(parser, top_level=True)
    sub = parser.add_subparsers(dest="command", metavar="<command>", title="commands")

    def add(name: str, help_text: str, description: str, epilog: str) -> argparse.ArgumentParser:
        p = sub.add_parser(
            name,
            help=help_text,
            description=description,
            epilog=epilog,
            formatter_class=_HelpFormatter,
        )
        _add_common(p, top_level=False)
        return p

    # demo ------------------------------------------------------------------
    p = add(
        "demo",
        "write synthetic approach frames (with masks) to try the tools",
        "Generate synthetic approach frames and their ground-truth masks.\n"
        "They are rendered, not photographs, and exist so you can try every command\n"
        "without downloading the dataset.",
        "examples:\n"
        "  runway-detection demo\n"
        "  runway-detection demo --output-dir /tmp/frames --count 8 --size 1920x1080",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo"),
        metavar="DIR",
        help="writes DIR/images/*.png and DIR/masks/*.png",
    )
    p.add_argument("--count", type=_positive_int, default=4, metavar="N", help="number of frames")
    p.add_argument(
        "--size",
        type=_size,
        default=(960, 540),
        metavar="WxH",
        help="frame size in pixels, e.g. 1920x1080",
    )
    p.set_defaults(func=_cmd_demo)

    # classical -------------------------------------------------------------
    p = add(
        "classical",
        "edge/shape-based detector (no model needed)",
        "Detect runway candidates with edges and shape rules. Needs no model or training\n"
        "data. Prints a JSON report to stdout (progress goes to stderr), or writes it\n"
        "with --report.",
        "examples:\n"
        "  runway-detection classical frame.png\n"
        "  runway-detection classical images/ --output-dir masks/ --overlay --report report.json\n"
        "  runway-detection classical images/ --preset basic -q > report.json\n\n"
        "exit status is 1 only if no input image could be read.",
    )
    p.add_argument("inputs", nargs="+", metavar="INPUT", help="image files and/or directories")
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="improved",
        help="detector settings: 'improved' scores candidates, 'basic' uses hard filters",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="write <name>_mask.png for every image with a detection",
    )
    p.add_argument(
        "--overlay",
        action="store_true",
        help="also write <name>_overlay.png with the detection drawn on the image "
        "(needs --output-dir)",
    )
    p.add_argument(
        "--report", type=Path, metavar="FILE", help="write the JSON report here instead of stdout"
    )
    p.set_defaults(func=_cmd_classical)

    # split -----------------------------------------------------------------
    p = add(
        "split",
        "create train/val/test folders from the dataset",
        "Copy the dataset into train/val/test folders. Everything in --train-dir becomes\n"
        "train; --test-dir is shuffled (seeded) and divided into val and test. With the\n"
        "dataset's own 70/30 split the default gives roughly 70/10/20.",
        "example:\n"
        "  runway-detection split --train-dir archive/.../train --test-dir archive/.../test \\\n"
        "      --output-dir data/split",
    )
    p.add_argument(
        "--train-dir", type=Path, required=True, metavar="DIR", help="original training images"
    )
    p.add_argument(
        "--test-dir", type=Path, required=True, metavar="DIR", help="original test images"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="creates DIR/train, DIR/val and DIR/test",
    )
    p.add_argument(
        "--val-fraction",
        type=_open_unit_interval,
        default=1 / 3,
        metavar="F",
        help="share of the original test set that becomes validation",
    )
    p.add_argument("--seed", type=int, default=42, metavar="N", help="shuffle seed")
    p.set_defaults(func=_cmd_split)

    # train -----------------------------------------------------------------
    p = add(
        "train",
        "train the U-Net segmentation model",
        "Train the U-Net on image/mask pairs (masks must have the same file names as\n"
        "the images; any non-zero pixel is runway). Keeps the best checkpoint by\n"
        "validation loss and writes metrics.json and training_curves.png.",
        "example:\n"
        "  runway-detection train --train-images data/split/train --train-masks labels/train \\\n"
        "      --val-images data/split/val --val-masks labels/test --epochs 50\n\n"
        "requires the ml extra: pip install -e '.[ml]'",
    )
    data = p.add_argument_group("data")
    data.add_argument(
        "--train-images", type=Path, required=True, metavar="DIR", help="training images"
    )
    data.add_argument(
        "--train-masks",
        type=Path,
        required=True,
        metavar="DIR",
        help="masks for the training images",
    )
    data.add_argument(
        "--val-images", type=Path, required=True, metavar="DIR", help="validation images"
    )
    data.add_argument(
        "--val-masks",
        type=Path,
        required=True,
        metavar="DIR",
        help="masks for the validation images",
    )
    data.add_argument(
        "--test-images",
        type=Path,
        metavar="DIR",
        help="optional held-out set evaluated with the best checkpoint",
    )
    data.add_argument("--test-masks", type=Path, metavar="DIR", help="masks for the test images")
    hyper = p.add_argument_group("training")
    hyper.add_argument(
        "--epochs", type=_positive_int, default=50, metavar="N", help="training epochs"
    )
    hyper.add_argument(
        "--batch-size", type=_positive_int, default=4, metavar="N", help="images per batch"
    )
    hyper.add_argument(
        "--lr", type=_positive_float, default=1e-4, metavar="RATE", help="Adam learning rate"
    )
    hyper.add_argument(
        "--image-size",
        type=_positive_int,
        default=512,
        metavar="PX",
        help="images are resized to PX x PX",
    )
    hyper.add_argument(
        "--num-workers",
        type=_non_negative_int,
        default=2,
        metavar="N",
        help="DataLoader worker processes (0 = load in the main process)",
    )
    hyper.add_argument("--seed", type=int, default=42, metavar="N", help="random seed")
    hyper.add_argument(
        "--device", metavar="NAME", help="e.g. cpu, cuda, cuda:1 (default: cuda if available)"
    )
    output = p.add_argument_group("output")
    output.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        metavar="DIR",
        help="checkpoint, metrics.json and curves go here",
    )
    output.add_argument("--no-plot", action="store_true", help="skip training_curves.png")
    p.set_defaults(func=_cmd_train)

    # predict ---------------------------------------------------------------
    p = add(
        "predict",
        "segment images with a trained U-Net",
        "Run a checkpoint from `train` on images at full resolution. Writes a mask per\n"
        "image, optional overlays, and predictions.json with coverage and regions.",
        "example:\n"
        "  runway-detection predict data/split/test --weights outputs/best_runway_model.pth \\\n"
        "      --output-dir predictions/ --overlay\n\n"
        "requires the ml extra: pip install -e '.[ml]'",
    )
    p.add_argument("inputs", nargs="+", metavar="INPUT", help="image files and/or directories")
    p.add_argument(
        "--weights",
        type=Path,
        required=True,
        metavar="FILE",
        help="checkpoint written by `train` (best_runway_model.pth)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("predictions"),
        metavar="DIR",
        help="masks, overlays and predictions.json go here",
    )
    p.add_argument(
        "--threshold",
        type=_open_unit_interval,
        default=0.5,
        metavar="P",
        help="probability above which a pixel counts as runway",
    )
    p.add_argument(
        "--image-size",
        type=_positive_int,
        default=512,
        metavar="PX",
        help="must match the size used for training",
    )
    p.add_argument("--device", metavar="NAME", help="e.g. cpu, cuda (default: cuda if available)")
    p.add_argument(
        "--overlay",
        action="store_true",
        help="also write <name>_overlay.png with runway pixels tinted red",
    )
    p.set_defaults(func=_cmd_predict)
    return parser


def _configure_logging(verbose: bool, quiet: bool) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_LogFormatter())
    log.handlers[:] = [handler]
    log.propagate = False
    log.setLevel(logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    _configure_logging(args.verbose, args.quiet)
    try:
        return int(args.func(args))
    except CliError as exc:
        log.error("%s", exc)
        return EXIT_ERROR
    except (FileNotFoundError, NotADirectoryError, ValueError, ImageLoadError) as exc:
        log.error("%s", exc)
        return EXIT_ERROR
    except ModuleNotFoundError as exc:
        log.error("%s. %s", exc, ML_HINT)
        return EXIT_ERROR
    except KeyboardInterrupt:
        log.error("interrupted")
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    raise SystemExit(main())
