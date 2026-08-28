from __future__ import annotations

import torch


def _validate_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if logits.shape != targets.shape:
        raise ValueError(
            "logits and targets must have identical shapes"
        )

    if logits.ndim != 4:
        raise ValueError(
            "logits and targets must have shape [B,C,H,W]"
        )


def segmentation_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Compute thresholded Dice coefficient.

    Args:
        logits: Raw model logits, [B, 1, H, W].
        targets: Binary ground-truth masks.
        threshold: Probability threshold used to create predictions.
        smooth: Numerical-stability constant.

    Returns:
        Mean Dice score across the batch.
    """

    _validate_inputs(
        logits,
        targets,
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    targets = targets.float()

    predictions = predictions.reshape(
        predictions.shape[0],
        -1,
    )

    targets = targets.reshape(
        targets.shape[0],
        -1,
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    denominator = (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return dice.mean()


def segmentation_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Compute thresholded intersection-over-union.

    Args:
        logits: Raw model logits, [B, 1, H, W].
        targets: Binary ground-truth masks.
        threshold: Probability threshold used to create predictions.
        smooth: Numerical-stability constant.

    Returns:
        Mean IoU score across the batch.
    """

    _validate_inputs(
        logits,
        targets,
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    targets = targets.float()

    predictions = predictions.reshape(
        predictions.shape[0],
        -1,
    )

    targets = targets.reshape(
        targets.shape[0],
        -1,
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)
    
    union = (
        predictions
        + targets
        - predictions * targets
    ).sum(dim=1)

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean()


def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute the standard CV-2 segmentation metrics.
    """

    dice = segmentation_dice(
        logits,
        targets,
        threshold=threshold,
    )

    iou = segmentation_iou(
        logits,
        targets,
        threshold=threshold,
    )

    return {
        "dice": float(dice.detach().cpu()),
        "iou": float(iou.detach().cpu()),
    }
