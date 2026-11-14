"""U-Net training loop with validation, checkpointing and a final test evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import RunwayDataset, build_transforms
from .model import ConfusionCounts, UNet, bce_dice_loss

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    train_images: Path
    train_masks: Path
    val_images: Path
    val_masks: Path
    test_images: Path | None = None
    test_masks: Path | None = None
    output_dir: Path = Path("outputs")
    epochs: int = 50
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-5
    image_size: int = 512
    num_workers: int = 2
    threshold: float = 0.5
    seed: int = 42
    device: str | None = None
    plot: bool = True


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5
) -> dict[str, float]:
    """Mean loss plus dataset-level IoU/precision/recall/F1."""
    model.eval()
    counts = ConfusionCounts()
    total_loss, batches = 0.0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_loss += float(bce_dice_loss(logits, masks))
            batches += 1
            counts.update(torch.sigmoid(logits) > threshold, masks > 0.5)
    return {"loss": total_loss / max(batches, 1), **counts.as_dict()}


def _loader(images: Path, masks: Path, cfg: TrainConfig, train: bool) -> DataLoader:
    dataset = RunwayDataset(images, masks, build_transforms(cfg.image_size, train=train))
    log.info("%s: %d image/mask pairs from %s", "train" if train else "eval", len(dataset), images)
    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train(cfg: TrainConfig) -> dict[str, object]:
    """Train a U-Net and return a summary; writes artefacts to ``cfg.output_dir``."""
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "best_runway_model.pth"
    log.info("Training on %s; artefacts go to %s", device, out)

    train_loader = _loader(cfg.train_images, cfg.train_masks, cfg, train=True)
    val_loader = _loader(cfg.val_images, cfg.val_masks, cfg, train=False)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    history: list[dict[str, float]] = []
    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for images, masks in tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = bce_dice_loss(model(images), masks)
            loss.backward()
            optimizer.step()
            running += loss.item()

        val = evaluate(model, val_loader, device, cfg.threshold)
        scheduler.step(val["loss"])
        row = {"epoch": epoch, "train_loss": running / len(train_loader), **_prefix(val, "val_")}
        row["lr"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        log.info(
            "epoch %d: train_loss=%.4f val_loss=%.4f val_iou=%.4f val_f1=%.4f",
            epoch,
            row["train_loss"],
            val["loss"],
            val["iou"],
            val["f1"],
        )
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(model.state_dict(), checkpoint)
            log.info("saved new best checkpoint (val_loss=%.4f)", best_val)

    summary: dict[str, object] = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "best_val_loss": best_val,
        "checkpoint": str(checkpoint),
        "history": history,
    }

    if cfg.test_images and cfg.test_masks:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        test_loader = _loader(cfg.test_images, cfg.test_masks, cfg, train=False)
        summary["test"] = evaluate(model, test_loader, device, cfg.threshold)
        log.info("test metrics: %s", summary["test"])

    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    if cfg.plot:
        plot_history(history, out / "training_curves.png")
    return summary


def _prefix(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{k}": v for k, v in metrics.items()}


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    """Save loss / IoU / F1 curves to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [h["val_iou"] for h in history])
    axes[1].set_title("Validation IoU")
    axes[2].plot(epochs, [h["val_f1"] for h in history])
    axes[2].set_title("Validation F1")
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
