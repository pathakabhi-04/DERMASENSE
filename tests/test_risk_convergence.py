"""
CV-8 risk convergence tests.

All synthetic -- constructs CandidateResult/TemporalResult directly
(same helper style as tests/test_pipeline_assembly.py) so the
escalation ratchet's logic is checked deterministically, no
checkpoints needed.
"""

from __future__ import annotations

from src.inference.orchestrator import CandidateResult
from src.risk.action_mapping import ProductAction
from src.risk.convergence import (
    MIN_TEMPORAL_CONFIDENCE_FOR_ESCALATION,
    RiskCategory,
    assess_risk,
)
from src.risk.safety_gate import GateDecision
from src.temporal.calibration import RulerCalibration
from src.temporal.delta import TemporalVerdict
from src.temporal.measurement import LesionMeasurement
from src.temporal.pipeline import TemporalResult


def _candidate(action: ProductAction, *, requires_review: bool = False) -> CandidateResult:
    return CandidateResult(
        candidate_index=0,
        box_pixels=(0, 0, 10, 10),
        detection_confidence=None,
        predicted_class="NEV",
        confidence=0.9,
        probabilities={"NEV": 0.9},
        product_action=action,
        gate_decision=GateDecision.REVIEW if requires_review else GateDecision.AUTO_RELEASE,
        requires_review=requires_review,
        gate_reason="test",
        mask_area_fraction=0.2,
        mask_degenerate=False,
        mask_touches_border=False,
        crop_blur=0.5,
        crop_contrast=0.5,
        calibrated_confidence=0.9,
    )


_DUMMY_MEASUREMENT = LesionMeasurement(True, "ok", 0.05, 1.1, (60.0, 10.0, 10.0), None, None)
_DUMMY_CALIBRATION = RulerCalibration(None, False, 0, "no ticks")


def _temporal(verdict: TemporalVerdict, *, magnitude: float = 1.5, confidence: float = 1.0) -> TemporalResult:
    return TemporalResult(
        verdict=verdict,
        magnitude=magnitude,
        confidence=confidence,
        per_feature_deltas={"size": None, "border": 0.1, "color": 5.0},
        compared_timestamps=("t1", "t2"),
        reason="test",
        earlier_measurement=_DUMMY_MEASUREMENT,
        later_measurement=_DUMMY_MEASUREMENT,
        earlier_calibration=_DUMMY_CALIBRATION,
        later_calibration=_DUMMY_CALIBRATION,
    )


# ---- base mapping (no temporal input) ----------------------------------


def test_base_mapping_monitor_is_low():
    result = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1")
    assert result.risk_category == RiskCategory.LOW


def test_base_mapping_evaluate_soon_is_medium():
    result = assess_risk(_candidate(ProductAction.EVALUATE_SOON), lesion_id="L1")
    assert result.risk_category == RiskCategory.MEDIUM


def test_base_mapping_urgent_is_high():
    result = assess_risk(_candidate(ProductAction.URGENT_EVALUATION), lesion_id="L1")
    assert result.risk_category == RiskCategory.HIGH


def test_base_mapping_unknown_is_high_failsafe():
    result = assess_risk(_candidate(ProductAction.UNKNOWN), lesion_id="L1")
    assert result.risk_category == RiskCategory.HIGH


def test_no_temporal_input_flags_no_comparison():
    result = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1", temporal=None)
    assert "NO_TEMPORAL_COMPARISON" in result.quality_flags
    assert result.temporal["verdict"] == TemporalVerdict.NO_PRIOR_DATA.value
    assert result.temporal["per_feature_deltas"]["size"] is None


# ---- escalation ratchet -------------------------------------------------


def test_growing_with_high_confidence_escalates_one_step():
    temporal = _temporal(TemporalVerdict.GROWING, magnitude=1.5, confidence=1.0)
    result = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1", temporal=temporal)

    assert result.risk_category == RiskCategory.MEDIUM
    assert result.requires_review is True
    assert "escalated" in result.risk_reason


def test_changed_color_with_high_confidence_escalates():
    temporal = _temporal(TemporalVerdict.CHANGED_COLOR, magnitude=1.2, confidence=1.0)
    result = assess_risk(_candidate(ProductAction.EVALUATE_SOON), lesion_id="L1", temporal=temporal)

    assert result.risk_category == RiskCategory.HIGH


def test_escalation_is_a_ceiling_not_wraparound():
    temporal = _temporal(TemporalVerdict.GROWING, magnitude=2.0, confidence=1.0)
    result = assess_risk(_candidate(ProductAction.URGENT_EVALUATION), lesion_id="L1", temporal=temporal)

    assert result.risk_category == RiskCategory.HIGH


def test_low_magnitude_does_not_escalate_even_if_verdict_growing():
    temporal = _temporal(TemporalVerdict.GROWING, magnitude=0.5, confidence=1.0)
    result = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1", temporal=temporal)

    assert result.risk_category == RiskCategory.LOW


def test_low_confidence_does_not_escalate_but_is_flagged():
    below_threshold = MIN_TEMPORAL_CONFIDENCE_FOR_ESCALATION - 0.01
    temporal = _temporal(TemporalVerdict.GROWING, magnitude=1.5, confidence=below_threshold)
    result = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1", temporal=temporal)

    assert result.risk_category == RiskCategory.LOW
    assert "TEMPORAL_LOW_CONFIDENCE" in result.quality_flags


def test_shrinking_never_deescalates_or_escalates():
    temporal = _temporal(TemporalVerdict.SHRINKING, magnitude=2.0, confidence=1.0)
    result = assess_risk(_candidate(ProductAction.EVALUATE_SOON), lesion_id="L1", temporal=temporal)

    assert result.risk_category == RiskCategory.MEDIUM  # unchanged from base


def test_stable_does_not_escalate():
    temporal = _temporal(TemporalVerdict.STABLE, magnitude=0.3, confidence=1.0)
    result = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1", temporal=temporal)

    assert result.risk_category == RiskCategory.LOW


def test_no_prior_data_does_not_escalate_and_is_flagged():
    temporal = _temporal(TemporalVerdict.NO_PRIOR_DATA, magnitude=0.0, confidence=0.0)
    result = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="L1", temporal=temporal)

    assert result.risk_category == RiskCategory.LOW
    assert "TEMPORAL_NO_PRIOR_DATA" in result.quality_flags


def test_escalation_forces_review_even_when_candidate_did_not_require_it():
    temporal = _temporal(TemporalVerdict.GROWING, magnitude=1.5, confidence=1.0)
    result = assess_risk(
        _candidate(ProductAction.MONITOR, requires_review=False), lesion_id="L1", temporal=temporal
    )

    assert result.requires_review is True


# ---- contract shape ------------------------------------------------------


def test_to_dict_matches_locked_contract_shape():
    temporal = _temporal(TemporalVerdict.STABLE)
    result = assess_risk(_candidate(ProductAction.MONITOR), lesion_id="lesion-42", temporal=temporal)
    payload = result.to_dict()

    assert set(payload.keys()) == {
        "lesion_id", "diagnosis", "risk_category", "risk_reason",
        "temporal", "uncertainty", "quality_flags",
    }
    assert set(payload["diagnosis"].keys()) == {"native_class", "probabilities"}
    assert set(payload["uncertainty"].keys()) == {"confidence", "requires_review"}
    assert payload["lesion_id"] == "lesion-42"
