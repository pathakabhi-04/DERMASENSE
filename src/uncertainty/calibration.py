"""
CV-6 calibration.

expected_calibration_error is generalized from
scripts/evaluate_c1_vs_f1_product.py::compute_ece (previously
script-local, no src/ dependency, moved here unchanged in logic).

Temperature scaling is applied post-hoc, on probabilities only:
calibrated = softmax(log(p) / T). This is mathematically equivalent to
the standard logit-temperature-scaling softmax(z / T) -- for any logits
z with softmax(z) = p, log(p) = z - logsumexp(z), and the unknown
per-example normalization constant logsumexp(z) cancels in the
softmax(log(p)/T) ratio. This means calibration needs no access to
NativePredictor's internal logits and requires no change to
src/inference/native.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fit once via scripts/calibrate_cv6_temperature.py against PAD-UFES val
# (n=336): raw ECE 0.0596 -> calibrated ECE 0.0401 at T=1.25. The model
# is mildly overconfident (T > 1 smooths the distribution). Not refit at
# pipeline construction time -- see docs/cv6_uncertainty_spec.md.
DEFAULT_TEMPERATURE = 1.25


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """
    Compute ECE and the per-bin breakdown.

    Args:
        probabilities: [N, C] array of class probabilities.
        y_true: [N] array of integer class indices.
        n_bins: number of equal-width confidence bins.

    Returns:
        (ece, per_bin_dataframe)
    """
    confidence = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correctness = (predictions == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = int(np.sum(mask))
        if count == 0:
            continue

        mean_confidence = float(np.mean(confidence[mask]))
        accuracy = float(np.mean(correctness[mask]))
        gap = abs(accuracy - mean_confidence)

        ece += (count / total) * gap

        rows.append(
            {
                "bin": i,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )

    return float(ece), pd.DataFrame(rows)


def apply_temperature(
    probabilities: np.ndarray, temperature: float
) -> np.ndarray:
    """
    Rescale probabilities by a temperature, equivalent to scaling the
    underlying (unobserved) logits by the same temperature. See module
    docstring for the derivation.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    log_p = np.log(np.clip(probabilities, 1e-12, 1.0))
    scaled = log_p / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)  # numerical stability
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=-1, keepdims=True)


def fit_temperature(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    candidates: np.ndarray | None = None,
    n_bins: int = 10,
) -> float:
    """
    Fit a single scalar temperature by grid search, minimizing ECE on a
    held-out labeled set (PAD-UFES val, per docs/cv6_uncertainty_spec.md).

    Grid search rather than gradient-based optimization: this is a
    one-off, bounded calibration fit on one validation set, not a
    training loop -- a transparent, inspectable search is preferable to
    adding an optimizer dependency for a 1-D problem.
    """
    if candidates is None:
        candidates = np.arange(0.5, 3.01, 0.05)

    best_temperature = 1.0
    best_ece = float("inf")

    for temperature in candidates:
        calibrated = apply_temperature(probabilities, float(temperature))
        ece, _ = expected_calibration_error(calibrated, y_true, n_bins=n_bins)
        if ece < best_ece:
            best_ece = ece
            best_temperature = float(temperature)

    return best_temperature
