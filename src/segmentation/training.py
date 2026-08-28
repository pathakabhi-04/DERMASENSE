from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.segmentation.losses import BCEDiceLoss
from src.segmentation.metrics import segmentation_metrics


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is preferable for the baseline.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(
    device: str | None = None,
) -> torch.device:
    """
    Resolve the requested device.

    If no device is specified, CUDA is preferred when available.
    """

    if device is None:
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    return resolved


def _move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move image and mask tensors to the training device."""

    images = batch["image"].to(
        device,
        non_blocking=True,
    )

    masks = batch["mask"].to(
        device,
        non_blocking=True,
    )

    return images, masks


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> float:
    """Train the model for one epoch."""

    model.train()

    running_loss = 0.0
    sample_count = 0

    for batch in loader:
        images, masks = _move_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        if use_amp:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
            ):
                logits = model(images)
                loss = criterion(
                    logits,
                    masks,
                )

            if scaler is None:
                raise RuntimeError(
                    "AMP requested without GradScaler."
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

            loss.backward()
            optimizer.step()

        batch_size = images.shape[0]

        running_loss += (
            loss.detach().item()
            * batch_size
        )

        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError(
            "Training loader produced no samples."
        )

    return running_loss / sample_count


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    threshold: float = 0.5,
    use_amp: bool = False,
) -> dict[str, float]:
    """
    Evaluate a model without updating parameters.

    This function is intended for validation and final test evaluation.
    """

    model.eval()

    running_loss = 0.0
    sample_count = 0

    dice_sum = 0.0
    iou_sum = 0.0

    for batch in loader:
        images, masks = _move_batch(
            batch,
            device,
        )

        if use_amp:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
            ):
                logits = model(images)
                loss = criterion(
                    logits,
                    masks,
                )
        else:
            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

        batch_size = images.shape[0]

        metrics = segmentation_metrics(
            logits,
            masks,
            threshold=threshold,
        )

        running_loss += (
            loss.item() * batch_size
        )

        dice_sum += (
            metrics["dice"] * batch_size
        )

        iou_sum += (
            metrics["iou"] * batch_size
        )

        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError(
            "Evaluation loader produced no samples."
        )

    return {
        "loss": running_loss / sample_count,
        "dice": dice_sum / sample_count,
        "iou": iou_sum / sample_count,
    }


def save_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_dice: float,
    history: list[dict[str, float]],
    config: dict[str, Any],
) -> None:
    """Save a complete resumable training checkpoint."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
        "history": history,
        "config": config,
    }

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load model and optionally optimizer state."""

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint


def save_history(
    history: list[dict[str, float]],
    path: str | Path,
) -> None:
    """Save training history as JSON."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            history,
            handle,
            indent=2,
        )


def fit(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module | None = None,
    scheduler: Any | None = None,
    epochs: int = 1,
    device: str | None = None,
    checkpoint_dir: str | Path = "checkpoints/cv2",
    config: dict[str, Any] | None = None,
    threshold: float = 0.5,
    use_amp: bool | None = None,
) -> list[dict[str, float]]:
    """
    Train a model using validation Dice for checkpoint selection.

    The test loader is intentionally absent from this API. This prevents
    accidental test-set usage during model development.
    """

    if epochs <= 0:
        raise ValueError(
            "epochs must be positive"
        )

    device_obj = resolve_device(device)

    model.to(device_obj)

    if criterion is None:
        criterion = BCEDiceLoss()

    if use_amp is None:
        use_amp = device_obj.type == "cuda"

    scaler = (
        torch.amp.GradScaler("cuda")
        if use_amp
        else None
    )

    checkpoint_dir = Path(
        checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history: list[dict[str, float]] = []

    best_val_dice = float("-inf")

    training_config = dict(
        config or {}
    )

    training_config.update(
        {
            "epochs": epochs,
            "device": str(device_obj),
            "threshold": threshold,
            "use_amp": use_amp,
        }
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device_obj,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device_obj,
            threshold=threshold,
            use_amp=use_amp,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(
                val_metrics["loss"]
            ),
            "val_dice": float(
                val_metrics["dice"]
            ),
            "val_iou": float(
                val_metrics["iou"]
            ),
            "learning_rate": float(
                current_lr
            ),
        }

        history.append(record)

        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_dice=max(
                best_val_dice,
                val_metrics["dice"],
            ),
            history=history,
            config=training_config,
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]

            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_dice=best_val_dice,
                history=history,
                config=training_config,
            )

        save_history(
            history,
            checkpoint_dir / "history.json",
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | "
            f"lr={current_lr:.6g}"
        )

    return history
