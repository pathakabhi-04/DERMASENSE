"""Losses and metrics, matching the CV-3 and CV-4 definitions on main."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn


class BCEDiceLoss(nn.Module):
    """0.5 BCE + 0.5 soft Dice (smooth 1), as src/segmentation/losses.py."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        probs = torch.sigmoid(logits).flatten(1)
        flat = targets.flatten(1)
        dice = (2 * (probs * flat).sum(1) + 1.0) / (probs.sum(1) + flat.sum(1) + 1.0)
        return 0.5 * self.bce(logits, targets) + 0.5 * (1 - dice.mean())


def sqrt_inverse_frequency_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    """CV-4 class weighting: sqrt(N / (K * n_c)), normalised to mean 1."""
    counts = torch.bincount(torch.tensor(targets), minlength=num_classes).double()
    if torch.any(counts == 0):
        raise ValueError("a class has no training samples")
    w = torch.sqrt(counts.sum() / (num_classes * counts))
    return (w / w.mean()).float()


@torch.no_grad()
def per_image_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Thresholded Dice per image (smooth 1), as src/segmentation/metrics.py."""
    pred = (torch.sigmoid(logits.float()) >= threshold).float().flatten(1)
    t = targets.float().flatten(1)
    return (2 * (pred * t).sum(1) + 1.0) / (pred.sum(1) + t.sum(1) + 1.0)


@torch.no_grad()
def per_image_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred = (torch.sigmoid(logits.float()) >= threshold).float().flatten(1)
    t = targets.float().flatten(1)
    inter = (pred * t).sum(1)
    return (inter + 1.0) / (pred.sum(1) + t.sum(1) - inter + 1.0)


def macro_f1(y_true, y_pred, num_classes: int) -> float:
    """Unweighted mean F1 over all classes; a class never seen nor predicted scores 0."""
    return float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0))


def per_class_f1(y_true, y_pred, num_classes: int) -> np.ndarray:
    return f1_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
