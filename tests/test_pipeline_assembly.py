"""
CV-1 -> CV-4 assembled pipeline tests.

Two layers:

1. Pure unit tests on the aggregation/outcome semantics. These encode
   the safety invariants from docs/cv1_cv4_assembly_spec.md (an
   unassessed image must never read as a cleared one, multi-candidate
   aggregation takes the most severe action) and need no checkpoints.
2. An integration test over real checkpoints and real images, asserting
   invariants rather than exact accuracy. Skipped when checkpoints are
   not present locally.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.inference.orchestrator import (
    CandidateResult,
    DermaSensePipeline,
    PipelineOutcome,
    PipelineResult,
    _candidate_lesion_id,
    _resolve_temporal_pairing,
)
from src.quality.assessment import assess_image
from src.risk.action_mapping import ProductAction
from src.risk.safety_gate import GateDecision

REPO_ROOT = Path(__file__).resolve().parents[1]

ROUTER_CHECKPOINT = REPO_ROOT / "checkpoints/cv1_5_router/best.pt"
SEGMENTATION_CHECKPOINT = REPO_ROOT / "checkpoints/cv3_512/best.pt"
CLASSIFIER_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
)
SECOND_CLASSIFIER_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed123_best.pt"
)
PAD_UFES_TEST = REPO_ROOT / "data/splits/pad_ufes/test.csv"

CHECKPOINTS_PRESENT = all(
    path.exists()
    for path in (
        ROUTER_CHECKPOINT,
        SEGMENTATION_CHECKPOINT,
        CLASSIFIER_CHECKPOINT,
    )
)
ENSEMBLE_CHECKPOINTS_PRESENT = (
    CHECKPOINTS_PRESENT and SECOND_CLASSIFIER_CHECKPOINT.exists()
)


def _candidate(
    action: ProductAction,
    *,
    requires_review: bool = False,
    index: int = 0,
) -> CandidateResult:
    return CandidateResult(
        candidate_index=index,
        box_pixels=(0, 0, 10, 10),
        detection_confidence=None,
        predicted_class="NEV",
        confidence=0.9,
        probabilities={"NEV": 0.9},
        product_action=action,
        gate_decision=(
            GateDecision.REVIEW if requires_review else GateDecision.AUTO_RELEASE
        ),
        requires_review=requires_review,
        gate_reason="test",
        mask_area_fraction=0.2,
        mask_degenerate=False,
        mask_touches_border=False,
        crop_blur=0.5,
        crop_contrast=0.5,
        calibrated_confidence=0.9,
    )


def _result(outcome: PipelineOutcome, candidates=()) -> PipelineResult:
    quality = assess_image(
        np.random.default_rng(0).integers(
            0, 256, size=(256, 256, 3), dtype=np.uint8
        )
    )
    return PipelineResult(
        outcome=outcome,
        quality=quality,
        framing="pre_framed",
        candidates=tuple(candidates),
    )


# ---- safety invariants (no checkpoints needed) -------------------


@pytest.mark.parametrize(
    "outcome",
    [PipelineOutcome.QUALITY_REJECTED, PipelineOutcome.NO_CANDIDATES],
)
def test_unassessed_image_is_never_auto_released(outcome):
    """An image the pipeline never assessed must not read as cleared."""
    result = _result(outcome)

    assert not result.assessed
    assert result.requires_review
    assert result.product_action is ProductAction.UNKNOWN


def test_no_candidates_is_distinct_from_assessed_low_risk():
    """The silent-miss concern: 'never looked' != 'looked and it's fine'."""
    missed = _result(PipelineOutcome.NO_CANDIDATES)
    cleared = _result(
        PipelineOutcome.ASSESSED, [_candidate(ProductAction.MONITOR)]
    )

    assert missed.outcome is not cleared.outcome
    assert missed.requires_review
    assert not missed.assessed and cleared.assessed


def test_multi_candidate_action_takes_most_severe():
    result = _result(
        PipelineOutcome.ASSESSED,
        [
            _candidate(ProductAction.MONITOR, index=0),
            _candidate(ProductAction.URGENT_EVALUATION, index=1),
            _candidate(ProductAction.EVALUATE_SOON, index=2),
        ],
    )

    assert result.product_action is ProductAction.URGENT_EVALUATION


def test_any_candidate_requiring_review_escalates_the_image():
    result = _result(
        PipelineOutcome.ASSESSED,
        [
            _candidate(ProductAction.MONITOR, index=0),
            _candidate(
                ProductAction.MONITOR, requires_review=True, index=1
            ),
        ],
    )

    assert result.requires_review


def test_unknown_never_outranks_a_real_action():
    result = _result(
        PipelineOutcome.ASSESSED,
        [
            _candidate(ProductAction.UNKNOWN, index=0),
            _candidate(ProductAction.MONITOR, index=1),
        ],
    )

    assert result.product_action is ProductAction.MONITOR


def test_result_is_serializable():
    result = _result(
        PipelineOutcome.ASSESSED, [_candidate(ProductAction.MONITOR)]
    )

    payload = result.to_dict()

    assert payload["outcome"] == "ASSESSED"
    assert payload["num_candidates"] == 1
    assert payload["candidates"][0]["predicted_class"] == "NEV"


# ---- CV-7/CV-8 pairing logic (no checkpoints needed) --------------


def test_no_prior_image_never_pairs():
    should_pair, reason = _resolve_temporal_pairing(1, None)
    assert not should_pair
    assert reason is None


def test_single_candidate_with_prior_image_pairs():
    should_pair, reason = _resolve_temporal_pairing(1, np.zeros((4, 4, 3)))
    assert should_pair
    assert reason is None


def test_multiple_candidates_with_prior_image_is_ambiguous_not_guessed():
    should_pair, reason = _resolve_temporal_pairing(3, np.zeros((4, 4, 3)))
    assert not should_pair
    assert reason == "PRIOR_IMAGE_PAIRING_AMBIGUOUS"


def test_zero_candidates_never_pairs():
    should_pair, _ = _resolve_temporal_pairing(0, np.zeros((4, 4, 3)))
    assert not should_pair


def test_lesion_id_passes_through_when_unambiguous():
    assert _candidate_lesion_id("L1", 0, 1) == "L1"


def test_lesion_id_suffixed_when_multiple_candidates():
    assert _candidate_lesion_id("L1", 0, 2) == "L1-0"
    assert _candidate_lesion_id("L1", 1, 2) == "L1-1"


def test_lesion_id_synthesized_when_none_given():
    assert _candidate_lesion_id(None, 2, 1) == "candidate-2"


# ---- integration over real checkpoints ---------------------------


@pytest.mark.skipif(
    not CHECKPOINTS_PRESENT, reason="component checkpoints not available"
)
def test_pipeline_runs_end_to_end_on_real_pad_ufes_images():
    pipeline = DermaSensePipeline.from_checkpoints(
        router_checkpoint=ROUTER_CHECKPOINT,
        segmentation_checkpoint=SEGMENTATION_CHECKPOINT,
        classifier_checkpoint=CLASSIFIER_CHECKPOINT,
        detector_weights=None,  # pre-framed branch only
        device="cpu",
    )

    rows = pd.read_csv(PAD_UFES_TEST).head(3)

    for _, row in rows.iterrows():
        image_bgr = cv2.imread(str(REPO_ROOT / row["image_path"]))
        assert image_bgr is not None

        result = pipeline.predict(image_bgr)

        assert isinstance(result, PipelineResult)
        assert result.outcome in set(PipelineOutcome)

        if result.assessed:
            assert result.candidates
            for candidate in result.candidates:
                assert candidate.predicted_class in {
                    "ACK",
                    "BCC",
                    "MEL",
                    "NEV",
                    "SCC",
                    "SEK",
                }
                assert 0.0 <= candidate.confidence <= 1.0
                assert (
                    abs(sum(candidate.probabilities.values()) - 1.0) < 1e-4
                )
                assert candidate.requires_review == (
                    candidate.gate_decision is GateDecision.REVIEW
                )
                assert 0.0 <= candidate.mask_area_fraction <= 1.0
                assert 0.0 <= candidate.crop_blur <= 1.0
                assert 0.0 <= candidate.crop_contrast <= 1.0
                assert 0.0 <= candidate.calibrated_confidence <= 1.0
                # Ensemble not requested for this pipeline instance.
                assert candidate.ensemble_agree is None


@pytest.mark.skipif(
    not ENSEMBLE_CHECKPOINTS_PRESENT,
    reason="ensemble checkpoints not available",
)
def test_ensemble_evidence_populated_when_requested():
    """CV-6 evidence appears only when explicitly requested (opt-in)."""
    pipeline = DermaSensePipeline.from_checkpoints(
        router_checkpoint=ROUTER_CHECKPOINT,
        segmentation_checkpoint=SEGMENTATION_CHECKPOINT,
        classifier_checkpoint=CLASSIFIER_CHECKPOINT,
        additional_ensemble_checkpoints=(SECOND_CLASSIFIER_CHECKPOINT,),
        detector_weights=None,
        device="cpu",
    )

    rows = pd.read_csv(PAD_UFES_TEST).head(2)
    for _, row in rows.iterrows():
        image_bgr = cv2.imread(str(REPO_ROOT / row["image_path"]))
        result = pipeline.predict(image_bgr)
        if not result.assessed:
            continue
        for candidate in result.candidates:
            assert candidate.ensemble_agree in (True, False)
            assert candidate.ensemble_probability_distance >= 0.0
            assert candidate.ensemble_confidence_spread >= 0.0


@pytest.mark.skipif(
    not CHECKPOINTS_PRESENT, reason="component checkpoints not available"
)
def test_unusable_image_stops_before_routing():
    """A quality rejection must terminate before CV-1.5/CV-2/CV-3/CV-4."""
    pipeline = DermaSensePipeline.from_checkpoints(
        router_checkpoint=ROUTER_CHECKPOINT,
        segmentation_checkpoint=SEGMENTATION_CHECKPOINT,
        classifier_checkpoint=CLASSIFIER_CHECKPOINT,
        detector_weights=None,
        device="cpu",
    )

    black = np.full((512, 512, 3), 2, dtype=np.uint8)

    result = pipeline.predict(black)

    assert result.outcome is PipelineOutcome.QUALITY_REJECTED
    assert result.framing is None
    assert result.candidates == ()
    assert result.requires_review


@pytest.mark.skipif(
    not CHECKPOINTS_PRESENT, reason="component checkpoints not available"
)
def test_risk_assessment_always_populated_without_prior_image():
    """CV-8 runs for every candidate even with no temporal history."""
    pipeline = DermaSensePipeline.from_checkpoints(
        router_checkpoint=ROUTER_CHECKPOINT,
        segmentation_checkpoint=SEGMENTATION_CHECKPOINT,
        classifier_checkpoint=CLASSIFIER_CHECKPOINT,
        detector_weights=None,
        device="cpu",
    )

    row = pd.read_csv(PAD_UFES_TEST).iloc[0]
    image_bgr = cv2.imread(str(REPO_ROOT / row["image_path"]))

    result = pipeline.predict(image_bgr, lesion_id="test-lesion-1")

    assert result.assessed
    candidate = result.candidates[0]
    assert candidate.risk_assessment is not None
    assert candidate.risk_assessment.risk_category.value in {"LOW", "MEDIUM", "HIGH"}
    assert candidate.risk_assessment.lesion_id == "test-lesion-1"
    assert "NO_TEMPORAL_COMPARISON" in candidate.risk_assessment.quality_flags
    assert candidate.risk_assessment.temporal["verdict"] == "NO_PRIOR_DATA"

    payload = candidate.to_dict()
    assert payload["risk_assessment"]["risk_category"] in {"LOW", "MEDIUM", "HIGH"}


@pytest.mark.skipif(
    not CHECKPOINTS_PRESENT, reason="component checkpoints not available"
)
def test_prior_image_wires_real_cv7_comparison():
    """
    A single-candidate image with a prior image actually runs CV-7,
    not just CV-4 -- asserting the temporal pipeline was genuinely
    exercised (a real, non-placeholder verdict/timestamp), not that a
    specific verdict came out (these are two unrelated PAD-UFES photos,
    not real visit pairs of the same lesion, so no particular verdict
    is expected -- only that the comparison ran).
    """
    pipeline = DermaSensePipeline.from_checkpoints(
        router_checkpoint=ROUTER_CHECKPOINT,
        segmentation_checkpoint=SEGMENTATION_CHECKPOINT,
        classifier_checkpoint=CLASSIFIER_CHECKPOINT,
        detector_weights=None,
        device="cpu",
    )

    rows = pd.read_csv(PAD_UFES_TEST).head(2)
    image_a = cv2.imread(str(REPO_ROOT / rows.iloc[0]["image_path"]))
    image_b = cv2.imread(str(REPO_ROOT / rows.iloc[1]["image_path"]))

    result = pipeline.predict(
        image_b,
        lesion_id="test-lesion-2",
        prior_image_bgr=image_a,
        prior_timestamp="2024-01-01",
        current_timestamp="2024-06-01",
    )

    assert result.assessed
    candidate = result.candidates[0]
    risk = candidate.risk_assessment
    assert risk is not None
    assert "NO_TEMPORAL_COMPARISON" not in risk.quality_flags
    assert risk.temporal["compared_timestamps"] == ["2024-01-01", "2024-06-01"]
    assert risk.temporal["verdict"] in {
        "STABLE", "GROWING", "SHRINKING", "CHANGED_COLOR", "NO_PRIOR_DATA",
    }
