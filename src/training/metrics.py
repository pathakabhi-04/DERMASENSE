from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


class MetricsError(ValueError):
    """Raised when invalid metric inputs are supplied."""


@dataclass(frozen=True)
class ClassificationMetrics:
    """
    Classification metrics for one evaluation pass.

    All values are scalar floats so the result is easy to log,
    serialize, and compare between experiments.
    """

    accuracy: float
    macro_f1: float
    weighted_f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
        }


def _validate_inputs(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    if not isinstance(predictions, torch.Tensor):
        raise MetricsError(
            "predictions must be a torch.Tensor."
        )

    if not isinstance(targets, torch.Tensor):
        raise MetricsError(
            "targets must be a torch.Tensor."
        )

    if num_classes <= 1:
        raise MetricsError(
            "num_classes must be greater than one."
        )

    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()

    if predictions.ndim == 2:
        if predictions.shape[1] != num_classes:
            raise MetricsError(
                "Logit shape does not match num_classes: "
                f"{tuple(predictions.shape)} vs "
                f"{num_classes} classes."
            )

        predictions = predictions.argmax(dim=1)

    elif predictions.ndim == 1:
        predictions = predictions.long()

    else:
        raise MetricsError(
            "predictions must have shape [N] or [N, C]. "
            f"Got {tuple(predictions.shape)}."
        )

    if targets.ndim != 1:
        raise MetricsError(
            "targets must have shape [N]. "
            f"Got {tuple(targets.shape)}."
        )

    if predictions.shape[0] != targets.shape[0]:
        raise MetricsError(
            "Prediction/target length mismatch: "
            f"{predictions.shape[0]} vs "
            f"{targets.shape[0]}."
        )

    if predictions.numel() == 0:
        raise MetricsError(
            "Cannot calculate metrics for an empty set."
        )

    if torch.any(targets < 0) or torch.any(
        targets >= num_classes
    ):
        raise MetricsError(
            "Targets contain values outside the valid "
            f"class range [0, {num_classes - 1}]."
        )

    if torch.any(predictions < 0) or torch.any(
        predictions >= num_classes
    ):
        raise MetricsError(
            "Predictions contain values outside the valid "
            f"class range [0, {num_classes - 1}]."
        )

    return predictions, targets.long()


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Build an integer confusion matrix.

    Rows    = true class
    Columns = predicted class
    """

    predictions, targets = _validate_inputs(
        predictions,
        targets,
        num_classes,
    )

    matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
    )

    for target, prediction in zip(
        targets.tolist(),
        predictions.tolist(),
    ):
        matrix[target, prediction] += 1

    return matrix


def accuracy_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """Calculate overall classification accuracy."""

    predictions, targets = _validate_inputs(
        predictions,
        targets,
        num_classes,
    )

    return float(
        (predictions == targets)
        .float()
        .mean()
        .item()
    )


def per_class_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Calculate F1 for every class.

    Classes with no true and no predicted examples receive 0.0.
    """

    matrix = confusion_matrix(
        predictions,
        targets,
        num_classes,
    ).float()

    true_positive = torch.diag(matrix)

    false_positive = (
        matrix.sum(dim=0)
        - true_positive
    )

    false_negative = (
        matrix.sum(dim=1)
        - true_positive
    )

    denominator = (
        2.0 * true_positive
        + false_positive
        + false_negative
    )

    f1 = torch.zeros(
        num_classes,
        dtype=torch.float32,
    )

    valid = denominator > 0

    f1[valid] = (
        2.0 * true_positive[valid]
        / denominator[valid]
    )

    return f1


def macro_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """Calculate unweighted mean F1 across all classes."""

    f1 = per_class_f1(
        predictions,
        targets,
        num_classes,
    )

    return float(f1.mean().item())


def weighted_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Calculate F1 weighted by the number of true examples
    in each class.
    """

    matrix = confusion_matrix(
        predictions,
        targets,
        num_classes,
    ).float()

    support = matrix.sum(dim=1)

    f1 = per_class_f1(
        predictions,
        targets,
        num_classes,
    )

    total = support.sum()

    if total == 0:
        raise MetricsError(
            "Cannot calculate weighted F1 for empty targets."
        )

    return float(
        (f1 * support).sum().div(total).item()
    )


def classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> ClassificationMetrics:
    """
    Calculate the standard Stage-1 classification metrics.
    """

    return ClassificationMetrics(
        accuracy=accuracy_score(
            predictions,
            targets,
            num_classes,
        ),
        macro_f1=macro_f1_score(
            predictions,
            targets,
            num_classes,
        ),
        weighted_f1=weighted_f1_score(
            predictions,
            targets,
            num_classes,
        ),
    )


def class_support(
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Return the number of true examples for each class."""

    if targets.ndim != 1:
        raise MetricsError(
            "targets must have shape [N]."
        )

    targets = targets.detach().cpu().long()

    if torch.any(targets < 0) or torch.any(
        targets >= num_classes
    ):
        raise MetricsError(
            "Targets contain invalid class indices."
        )

    return torch.bincount(
        targets,
        minlength=num_classes,
    )


def per_class_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> list[dict[str, float]]:
    """
    Return precision, recall, F1, and support for every class.
    """

    matrix = confusion_matrix(
        predictions,
        targets,
        num_classes,
    ).float()

    true_positive = torch.diag(matrix)

    false_positive = (
        matrix.sum(dim=0)
        - true_positive
    )

    false_negative = (
        matrix.sum(dim=1)
        - true_positive
    )

    support = matrix.sum(dim=1)

    precision_denominator = (
        true_positive + false_positive
    )

    recall_denominator = (
        true_positive + false_negative
    )

    precision = torch.zeros(
        num_classes,
        dtype=torch.float32,
    )

    recall = torch.zeros(
        num_classes,
        dtype=torch.float32,
    )

    precision_valid = precision_denominator > 0
    recall_valid = recall_denominator > 0

    precision[precision_valid] = (
        true_positive[precision_valid]
        / precision_denominator[precision_valid]
    )

    recall[recall_valid] = (
        true_positive[recall_valid]
        / recall_denominator[recall_valid]
    )

    f1 = per_class_f1(
        predictions,
        targets,
        num_classes,
    )

    result = []

    for index in range(num_classes):
        result.append(
            {
                "class_index": float(index),
                "precision": float(
                    precision[index].item()
                ),
                "recall": float(
                    recall[index].item()
                ),
                "f1": float(
                    f1[index].item()
                ),
                "support": float(
                    support[index].item()
                ),
            }
        )

    return result
