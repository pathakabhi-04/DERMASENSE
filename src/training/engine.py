from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from src.training.metrics import ClassificationMetrics, classification_metrics


class TrainingError(RuntimeError):
    """Raised when training configuration or execution is invalid."""


@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for one native-diagnosis training run.
    """

    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cpu"
    gradient_clip_norm: Optional[float] = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise TrainingError(
                "epochs must be greater than zero."
            )

        if self.learning_rate <= 0:
            raise TrainingError(
                "learning_rate must be greater than zero."
            )

        if self.weight_decay < 0:
            raise TrainingError(
                "weight_decay cannot be negative."
            )

        if self.gradient_clip_norm is not None:
            if self.gradient_clip_norm <= 0:
                raise TrainingError(
                    "gradient_clip_norm must be greater than zero."
                )


@dataclass(frozen=True)
class EpochResult:
    """
    Result of one training or validation epoch.
    """

    loss: float
    metrics: ClassificationMetrics

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            **self.metrics.as_dict(),
        }


@dataclass(frozen=True)
class TrainingHistory:
    """
    Complete history for a training run.
    """

    train: tuple[EpochResult, ...]
    val: tuple[EpochResult, ...]

    @property
    def best_epoch(self) -> int:
        if not self.val:
            raise TrainingError(
                "Cannot determine best epoch from empty validation history."
            )

        return max(
            range(len(self.val)),
            key=lambda index: self.val[index].metrics.macro_f1,
        ) + 1

    @property
    def best_val_macro_f1(self) -> float:
        if not self.val:
            raise TrainingError(
                "Cannot determine best validation score from empty history."
            )

        return max(
            result.metrics.macro_f1
            for result in self.val
        )


def _resolve_device(device: str) -> torch.device:
    """
    Resolve and validate the requested torch device.
    """

    if device == "cuda":
        if not torch.cuda.is_available():
            raise TrainingError(
                "CUDA was requested but is not available."
            )

    if device == "mps":
        if not torch.backends.mps.is_available():
            raise TrainingError(
                "MPS was requested but is not available."
            )

    return torch.device(device)


def _extract_batch(
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract image tensors and native target indices.
    """

    if "image" not in batch:
        raise TrainingError(
            "Batch is missing required key: 'image'."
        )

    if "target" not in batch:
        raise TrainingError(
            "Batch is missing required key: 'target'."
        )

    images = batch["image"]
    targets = batch["target"]

    if not isinstance(images, torch.Tensor):
        raise TrainingError(
            "Batch 'image' must be a torch.Tensor."
        )

    if not isinstance(targets, torch.Tensor):
        raise TrainingError(
            "Batch 'target' must be a torch.Tensor."
        )

    if images.ndim != 4:
        raise TrainingError(
            "Expected images with shape [B, C, H, W]. "
            f"Got {tuple(images.shape)}."
        )

    if targets.ndim != 1:
        raise TrainingError(
            "Expected targets with shape [B]. "
            f"Got {tuple(targets.shape)}."
        )

    if images.shape[0] != targets.shape[0]:
        raise TrainingError(
            "Image/target batch size mismatch."
        )

    return images, targets.long()


def _validate_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> None:
    """
    Validate model output before loss calculation.
    """

    if not isinstance(logits, torch.Tensor):
        raise TrainingError(
            "Model output must be a torch.Tensor."
        )

    if logits.ndim != 2:
        raise TrainingError(
            "Expected logits with shape [B, C]. "
            f"Got {tuple(logits.shape)}."
        )

    if logits.shape[0] != targets.shape[0]:
        raise TrainingError(
            "Logit/target batch size mismatch."
        )

    if logits.shape[1] != num_classes:
        raise TrainingError(
            "Model output class count does not match "
            f"dataset target space: {logits.shape[1]} "
            f"vs {num_classes}."
        )

    if not torch.isfinite(logits).all():
        raise TrainingError(
            "Model produced non-finite logits."
        )


def _run_epoch(
    *,
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    num_classes: int,
    dataset_id: str,
    training: bool,
    gradient_clip_norm: Optional[float],
) -> EpochResult:
    """
    Run one training or validation epoch.
    """

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in loader:
        images, targets = _extract_batch(batch)

        images = images.to(device)
        targets = targets.to(device)

        if training:
            if optimizer is None:
                raise TrainingError(
                    "Training requires an optimizer."
                )

            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(
                images,
                dataset_id=dataset_id,
            )

            _validate_logits(
                logits,
                targets,
                num_classes,
            )

            loss = criterion(
                logits,
                targets,
            )

            if not torch.isfinite(loss):
                raise TrainingError(
                    "Loss became non-finite."
                )

            if training:
                loss.backward()

                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        gradient_clip_norm,
                    )

                optimizer.step()

        batch_size = targets.shape[0]

        total_loss += (
            loss.detach().item()
            * batch_size
        )

        total_samples += batch_size

        predictions = logits.detach().argmax(dim=1)

        all_predictions.append(
            predictions.cpu()
        )

        all_targets.append(
            targets.detach().cpu()
        )

    if total_samples == 0:
        raise TrainingError(
            "Epoch processed zero samples."
        )

    predictions = torch.cat(
        all_predictions,
        dim=0,
    )

    targets = torch.cat(
        all_targets,
        dim=0,
    )

    metrics = classification_metrics(
        predictions,
        targets,
        num_classes,
    )

    return EpochResult(
        loss=total_loss / total_samples,
        metrics=metrics,
    )


class Trainer:
    """
    Reusable training engine for native diagnosis models.

    The trainer operates on one dataset_id at a time.

    The test split is intentionally not accepted here.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader,
        val_loader,
        dataset_id: str,
        num_classes: int,
        config: TrainingConfig | None = None,
        criterion: nn.Module | None = None,
        optimizer_factory: Optional[
            Callable[[list[nn.Parameter]], torch.optim.Optimizer]
        ] = None,
    ) -> None:

        if config is None:
            config = TrainingConfig()

        self.config = config
        self.dataset_id = dataset_id
        self.num_classes = num_classes

        self.device = _resolve_device(
            config.device
        )

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = (
            criterion
            if criterion is not None
            else nn.CrossEntropyLoss()
        )

        self.criterion = self.criterion.to(
            self.device
        )

        if optimizer_factory is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer_factory(
                list(self.model.parameters())
            )

    def train_epoch(self) -> EpochResult:
        """
        Run one training epoch.
        """

        return _run_epoch(
            model=self.model,
            loader=self.train_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            num_classes=self.num_classes,
            dataset_id=self.dataset_id,
            training=True,
            gradient_clip_norm=self.config.gradient_clip_norm,
        )

    @torch.no_grad()
    def validate_epoch(self) -> EpochResult:
        """
        Run one validation epoch.
        """

        return _run_epoch(
            model=self.model,
            loader=self.val_loader,
            criterion=self.criterion,
            optimizer=None,
            device=self.device,
            num_classes=self.num_classes,
            dataset_id=self.dataset_id,
            training=False,
            gradient_clip_norm=None,
        )

    def fit(
        self,
        *,
        checkpoint_path: str | None = None,
    ) -> TrainingHistory:
        """
        Train for the configured number of epochs.

        Best checkpoint is selected using validation macro-F1.
        The test set is never touched.
        """

        train_history: list[EpochResult] = []
        val_history: list[EpochResult] = []

        best_macro_f1 = float("-inf")

        for epoch in range(
            1,
            self.config.epochs + 1,
        ):
            train_result = self.train_epoch()
            val_result = self.validate_epoch()

            train_history.append(
                train_result
            )

            val_history.append(
                val_result
            )

            print(
                f"Epoch {epoch:03d}/{self.config.epochs:03d} | "
                f"train_loss={train_result.loss:.4f} | "
                f"train_macro_f1="
                f"{train_result.metrics.macro_f1:.4f} | "
                f"val_loss={val_result.loss:.4f} | "
                f"val_macro_f1="
                f"{val_result.metrics.macro_f1:.4f}"
            )

            if (
                checkpoint_path is not None
                and val_result.metrics.macro_f1
                > best_macro_f1
            ):
                best_macro_f1 = (
                    val_result.metrics.macro_f1
                )

                self.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    val_result=val_result,
                )

        return TrainingHistory(
            train=tuple(train_history),
            val=tuple(val_history),
        )

    def save_checkpoint(
        self,
        path: str,
        *,
        epoch: int,
        val_result: EpochResult,
    ) -> None:
        """
        Save a reproducible training checkpoint.
        """

        checkpoint = {
            "epoch": epoch,
            "dataset_id": self.dataset_id,
            "num_classes": self.num_classes,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "val_result": val_result.as_dict(),
        }

        torch.save(
            checkpoint,
            path,
        )

    def load_checkpoint(
        self,
        path: str,
    ) -> dict:
        """
        Load model and optimizer state from a checkpoint.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        required_keys = {
            "epoch",
            "dataset_id",
            "num_classes",
            "model_state_dict",
            "optimizer_state_dict",
        }

        missing = required_keys - checkpoint.keys()

        if missing:
            raise TrainingError(
                "Checkpoint is missing required keys: "
                f"{sorted(missing)}"
            )

        if checkpoint["dataset_id"] != self.dataset_id:
            raise TrainingError(
                "Checkpoint dataset_id does not match "
                f"trainer dataset_id: "
                f"{checkpoint['dataset_id']!r} vs "
                f"{self.dataset_id!r}."
            )

        if checkpoint["num_classes"] != self.num_classes:
            raise TrainingError(
                "Checkpoint num_classes does not match "
                "trainer configuration."
            )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        return checkpoint