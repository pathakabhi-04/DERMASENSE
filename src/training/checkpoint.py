from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is invalid or incompatible."""


@dataclass(frozen=True)
class CheckpointMetadata:
    architecture: str
    dataset_id: str
    num_classes: int
    epoch: int
    val_macro_f1: float


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    dataset_id: str,
    num_classes: int,
    architecture: str,
    val_macro_f1: float,
    config: dict[str, Any] | None = None,
) -> Path:
    """
    Save a complete training checkpoint.

    The checkpoint contains:
      - model weights
      - optimizer state
      - experiment metadata
      - optional configuration
    """

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if epoch < 0:
        raise CheckpointError(
            "epoch must be non-negative"
        )

    if num_classes <= 0:
        raise CheckpointError(
            "num_classes must be positive"
        )

    checkpoint = {
        "checkpoint_version": 1,
        "architecture": architecture,
        "dataset_id": dataset_id,
        "num_classes": num_classes,
        "epoch": epoch,
        "val_macro_f1": float(val_macro_f1),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),
        "config": config,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    expected_dataset_id: str | None = None,
    expected_num_classes: int | None = None,
    expected_architecture: str | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointMetadata:
    """
    Load a checkpoint into a model and optionally an optimizer.

    Dataset, class-count, and architecture compatibility can
    be checked before loading the weights.
    """

    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise CheckpointError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise CheckpointError(
            "Checkpoint must contain a dictionary."
        )

    required_keys = {
        "checkpoint_version",
        "architecture",
        "dataset_id",
        "num_classes",
        "epoch",
        "val_macro_f1",
        "model_state_dict",
    }

    missing = required_keys.difference(
        checkpoint.keys()
    )

    if missing:
        raise CheckpointError(
            "Checkpoint is missing required fields: "
            f"{sorted(missing)}"
        )

    if (
        expected_dataset_id is not None
        and checkpoint["dataset_id"]
        != expected_dataset_id
    ):
        raise CheckpointError(
            "Dataset mismatch: "
            f"checkpoint={checkpoint['dataset_id']!r}, "
            f"expected={expected_dataset_id!r}"
        )

    if (
        expected_num_classes is not None
        and checkpoint["num_classes"]
        != expected_num_classes
    ):
        raise CheckpointError(
            "Class-count mismatch: "
            f"checkpoint={checkpoint['num_classes']}, "
            f"expected={expected_num_classes}"
        )

    if (
        expected_architecture is not None
        and checkpoint["architecture"]
        != expected_architecture
    ):
        raise CheckpointError(
            "Architecture mismatch: "
            f"checkpoint={checkpoint['architecture']!r}, "
            f"expected={expected_architecture!r}"
        )

    try:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    except RuntimeError as exc:
        raise CheckpointError(
            "Model weights are incompatible with "
            "the supplied model."
        ) from exc

    optimizer_state = checkpoint.get(
        "optimizer_state_dict"
    )

    if optimizer is not None:
        if optimizer_state is None:
            raise CheckpointError(
                "Checkpoint does not contain optimizer "
                "state, but an optimizer was supplied."
            )

        optimizer.load_state_dict(
            optimizer_state
        )

    return CheckpointMetadata(
        architecture=checkpoint["architecture"],
        dataset_id=checkpoint["dataset_id"],
        num_classes=int(
            checkpoint["num_classes"]
        ),
        epoch=int(
            checkpoint["epoch"]
        ),
        val_macro_f1=float(
            checkpoint["val_macro_f1"]
        ),
    )


def inspect_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> CheckpointMetadata:
    """
    Read checkpoint metadata without modifying a model.
    """

    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise CheckpointError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise CheckpointError(
            "Checkpoint must contain a dictionary."
        )

    required_keys = {
        "checkpoint_version",
        "architecture",
        "dataset_id",
        "num_classes",
        "epoch",
        "val_macro_f1",
        "model_state_dict",
    }

    missing = required_keys.difference(
        checkpoint.keys()
    )

    if missing:
        raise CheckpointError(
            "Checkpoint is missing required fields: "
            f"{sorted(missing)}"
        )

    return CheckpointMetadata(
        architecture=checkpoint["architecture"],
        dataset_id=checkpoint["dataset_id"],
        num_classes=int(
            checkpoint["num_classes"]
        ),
        epoch=int(
            checkpoint["epoch"]
        ),
        val_macro_f1=float(
            checkpoint["val_macro_f1"]
        ),
    )