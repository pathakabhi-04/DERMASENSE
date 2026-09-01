"""
CV-6 ensemble disagreement.

Two independently-trained checkpoints (seed42, seed123) already exist
for the same architecture and training recipe -- this is the cheapest
possible version of an ensemble uncertainty signal, since the second
model requires no new training. See docs/cv6_uncertainty_spec.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.inference.native import NativePredictor, NativePrediction

DEFAULT_ENSEMBLE_CHECKPOINTS = (
    "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt",
    "checkpoints/archive/pad_ufes_c1_partial_finetune_seed123_best.pt",
)


def load_ensemble(
    checkpoint_paths: tuple[str | Path, ...] = DEFAULT_ENSEMBLE_CHECKPOINTS,
    *,
    device: str | torch.device = "cpu",
) -> list[NativePredictor]:
    """Load each ensemble member via the existing NativePredictor API."""
    return [
        NativePredictor.from_checkpoint(path, device=device)
        for path in checkpoint_paths
    ]


def ensemble_evidence(predictions: list[NativePrediction]) -> dict[str, Any]:
    """
    Summarize agreement/disagreement across ensemble member predictions.

    Evidence only -- does not decide anything. See the "evidence, not a
    decision" principle in docs/cv6_uncertainty_spec.md.
    """
    if len(predictions) < 2:
        raise ValueError("ensemble_evidence requires at least 2 predictions")

    class_names = sorted(predictions[0].probabilities.keys())
    prob_vectors = [
        np.array([p.probabilities[name] for name in class_names])
        for p in predictions
    ]

    predicted_classes = {p.predicted_class for p in predictions}
    agree = len(predicted_classes) == 1

    # Mean pairwise L1 distance between probability vectors -- 0 when
    # all members agree exactly, up to 2 when they disagree completely.
    pairwise_distances = []
    for i in range(len(prob_vectors)):
        for j in range(i + 1, len(prob_vectors)):
            pairwise_distances.append(
                float(np.abs(prob_vectors[i] - prob_vectors[j]).sum())
            )
    mean_probability_distance = float(np.mean(pairwise_distances))

    confidences = [p.confidence for p in predictions]

    return {
        "ensemble_agree": bool(agree),
        "ensemble_probability_distance": mean_probability_distance,
        "ensemble_confidence_spread": float(
            max(confidences) - min(confidences)
        ),
    }
