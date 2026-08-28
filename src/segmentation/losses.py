from __future__ import annotations

import torch
import torch.nn as nn


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Compute soft Dice score from logits and binary targets.

    Args:
        logits: [B, 1, H, W] raw model logits.
        targets: [B, 1, H, W] binary masks.

    Returns:
        Scalar Dice score averaged across the batch.
    """

    if logits.shape != targets.shape:
        raise ValueError(
            "logits and targets must have identical shapes"
        )

    if logits.ndim != 4:
        raise ValueError(
            "logits and targets must have shape [B,C,H,W]"
        )

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.reshape(
        probabilities.shape[0],
        -1,
    )

    targets = targets.float().reshape(
        targets.shape[0],
        -1,
    )

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return dice.mean()


class DiceLoss(nn.Module):
    """1 - soft Dice score."""

    def __init__(
        self,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        if smooth <= 0:
            raise ValueError(
                "smooth must be positive"
            )

        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return 1.0 - dice_score(
            logits,
            targets,
            smooth=self.smooth,
        )


class BCEDiceLoss(nn.Module):
    """
    Combined BCE-with-logits and Dice loss.

    total = bce_weight * BCE + dice_weight * Dice
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        if bce_weight < 0:
            raise ValueError(
                "bce_weight must be non-negative"
            )

        if dice_weight < 0:
            raise ValueError(
                "dice_weight must be non-negative"
            )

        if bce_weight + dice_weight <= 0:
            raise ValueError(
                "at least one loss weight must be positive"
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss(
            smooth=smooth,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        if logits.shape != targets.shape:
            raise ValueError(
                "logits and targets must have identical shapes"
            )

        targets = targets.float()

        bce = self.bce(
            logits,
            targets,
        )

        dice = self.dice(
            logits,
            targets,
        )

        return (
            self.bce_weight * bce
            + self.dice_weight * dice
        )
