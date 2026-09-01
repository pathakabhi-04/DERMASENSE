"""
CV-6 uncertainty unit tests.

Pure functions -- no checkpoints required. See tests/test_pipeline_assembly.py
for the real-checkpoint ensemble integration test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.uncertainty.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)
from src.uncertainty.ensemble import ensemble_evidence
from src.inference.native import NativePrediction
from src.risk.action_mapping import ProductAction
from src.risk.safety_gate import GateDecision, SafetyGateResult


def _prediction(predicted_class: str, probabilities: dict) -> NativePrediction:
    return NativePrediction(
        predicted_class=predicted_class,
        confidence=probabilities[predicted_class],
        probabilities=probabilities,
        product_action=ProductAction.MONITOR,
        safety_gate=SafetyGateResult(
            predicted_diagnosis=predicted_class,
            predicted_action=ProductAction.MONITOR,
            decision=GateDecision.AUTO_RELEASE,
            reason="test",
        ),
    )


# ---- calibration --------------------------------------------------


def test_apply_temperature_identity_at_t_equals_1():
    probs = np.array([[0.7, 0.2, 0.1]])
    calibrated = apply_temperature(probs, 1.0)
    assert np.allclose(calibrated, probs, atol=1e-6)


def test_apply_temperature_above_1_reduces_max_confidence():
    """T > 1 smooths the distribution -- the top class gets less peaky."""
    probs = np.array([[0.9, 0.05, 0.05]])
    calibrated = apply_temperature(probs, 2.0)
    assert calibrated.max() < probs.max()


def test_apply_temperature_rejects_nonpositive():
    with pytest.raises(ValueError):
        apply_temperature(np.array([[0.5, 0.5]]), 0.0)


def test_apply_temperature_rows_sum_to_one():
    probs = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]])
    calibrated = apply_temperature(probs, 1.7)
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_expected_calibration_error_perfect_calibration_is_zero():
    # confidence exactly matches accuracy in every bin
    rng = np.random.default_rng(0)
    n = 1000
    probs = np.column_stack(
        [np.full(n, 0.7), np.full(n, 0.3)]
    )
    # exactly 70% correct, matching confidence
    y_true = np.zeros(n, dtype=int)
    y_true[: int(n * 0.3)] = 1  # 30% wrong -> 70% accuracy at 0.7 confidence

    ece, per_bin = expected_calibration_error(probs, y_true, n_bins=10)
    assert ece < 0.02
    assert "mean_confidence" in per_bin.columns


def test_fit_temperature_improves_or_matches_ece_on_overconfident_data():
    """Synthetic overconfident predictor: T>1 should reduce ECE."""
    rng = np.random.default_rng(1)
    n = 500
    # Model is confidently wrong 30% of the time at confidence ~0.95
    y_true = rng.integers(0, 2, size=n)
    predicted = rng.integers(0, 2, size=n)
    correct_mask = predicted == y_true
    conf = np.where(correct_mask, 0.95, 0.95)  # always confident
    probs = np.zeros((n, 2))
    for i in range(n):
        probs[i, predicted[i]] = conf[i]
        probs[i, 1 - predicted[i]] = 1 - conf[i]

    raw_ece, _ = expected_calibration_error(probs, y_true)
    t = fit_temperature(probs, y_true)
    calibrated = apply_temperature(probs, t)
    calibrated_ece, _ = expected_calibration_error(calibrated, y_true)

    assert calibrated_ece <= raw_ece + 1e-9


# ---- ensemble -------------------------------------------------------


def test_ensemble_evidence_full_agreement():
    predictions = [
        _prediction("MEL", {"MEL": 0.8, "NEV": 0.2}),
        _prediction("MEL", {"MEL": 0.75, "NEV": 0.25}),
    ]
    evidence = ensemble_evidence(predictions)
    assert evidence["ensemble_agree"] is True
    assert evidence["ensemble_probability_distance"] == pytest.approx(0.1, abs=1e-6)


def test_ensemble_evidence_disagreement():
    predictions = [
        _prediction("MEL", {"MEL": 0.9, "NEV": 0.1}),
        _prediction("NEV", {"MEL": 0.1, "NEV": 0.9}),
    ]
    evidence = ensemble_evidence(predictions)
    assert evidence["ensemble_agree"] is False
    assert evidence["ensemble_probability_distance"] == pytest.approx(1.6, abs=1e-6)


def test_ensemble_evidence_requires_at_least_two():
    with pytest.raises(ValueError):
        ensemble_evidence([_prediction("MEL", {"MEL": 1.0})])


def test_ensemble_confidence_spread():
    predictions = [
        _prediction("MEL", {"MEL": 0.6, "NEV": 0.4}),
        _prediction("MEL", {"MEL": 0.95, "NEV": 0.05}),
    ]
    evidence = ensemble_evidence(predictions)
    assert evidence["ensemble_confidence_spread"] == pytest.approx(0.35, abs=1e-6)
