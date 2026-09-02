"""
CV-8 risk convergence: CV-4 + CV-6 + CV-7 -> one risk assessment.

This is the convergence point every prior CV-5/CV-6/CV-7 doc has
referred to but not implemented: CV-5/CV-6/CV-7 feed in as evidence,
CV-8 is the only place allowed to turn that evidence into a product
decision (docs/cv5_cv6_evidence_architecture.md's dependency-direction
principle, applied here to CV-7 for the first time).

Produces the exact JSON contract locked in
docs/cv7_temporal_rag_integration_spec.md:
`{lesion_id, diagnosis, risk_category, risk_reason, temporal,
uncertainty, quality_flags}`.

Reuses existing evidence types rather than re-deriving them:
`CandidateResult` (src/inference/orchestrator.py) already carries
CV-4's diagnosis/action and CV-6's calibrated confidence;
`TemporalResult` (src/temporal/pipeline.py) already carries CV-7's
verdict. This module's only new logic is how they combine.

## Two discrepancies from the locked spec, resolved here (not silently)

1. **`risk_category` (LOW|MEDIUM|HIGH) vs. `ProductAction`.** Kept as
   two separate things, not unified: `ProductAction` remains the
   internal, authoritative action used unchanged by the existing
   safety gate (`src/risk/safety_gate.py`) and orchestrator.
   `risk_category` is a NEW field this module derives from it, for the
   external contract only. Mapping: `URGENT_EVALUATION -> HIGH`,
   `EVALUATE_SOON -> MEDIUM`, `MONITOR -> LOW`, `UNKNOWN -> HIGH`
   (fail-safe -- an action the system couldn't resolve must never
   present as low risk to a downstream consumer, mirroring
   `safety_gate.py`'s own fail-to-REVIEW handling of UNKNOWN).
2. **`native_class` taxonomy (ISIC 8-class vs. PAD-UFES 6-class).**
   Left UNRESOLVED here, deliberately: this module passes through
   whatever `CandidateResult.predicted_class` actually is (PAD-UFES
   6-class, per `src/inference/native.py`) rather than fabricating a
   mapping to the contract's 8-class example. That's a CV-4/data
   taxonomy question, orthogonal to wiring CV-7's signal into risk
   convergence, and stays open per the spec's own note.

## `quality_flags`: surfacing evidence CV-1/CV-3/CV-6 already compute

`CandidateResult` already carries mask evidence (CV-3), crop-quality
evidence (CV-1's signals, applied to the crop -- CV-4 domain evidence
work), and ensemble evidence (CV-6) -- none of it reached this
contract's `quality_flags` before now. Disclosure only, same as every
other evidence field in this project: none of these ever change
`risk_category` or `requires_review` (only CV-7's escalation rule does
that, deliberately, per the section above).

- `DEGENERATE_MASK` / `MASK_TOUCHES_BORDER` -- direct passthrough of
  `candidate.mask_degenerate` / `mask_touches_border` (already boolean,
  no threshold to invent).
- `LOW_CROP_CONTRAST` -- `crop_contrast < 0.20`, reusing the cutoff
  independently validated in `docs/cv4_domain_evidence_spec.md` (BCC/ACK
  crops: 58.7%/5.1% fall below it vs. MEL's 5.1%/0.337 mean) -- a real,
  evidence-grounded discriminator, not a guess.
- `LOW_CROP_BLUR` -- `crop_blur < 0.15`, reusing
  `src/quality/assessment.py`'s whole-image advisory `blur_threshold`.
  Unlike contrast, blur was never independently calibrated at CROP
  scale (the CV-4 investigation implicated contrast, not blur) -- this
  is a borrowed threshold, flagged here as weaker evidence than
  `LOW_CROP_CONTRAST`, not claimed as equally validated.
- `ENSEMBLE_DISAGREEMENT` -- `ensemble_agree is False` (only meaningful
  when the ensemble ran at all; `None` means it didn't, and stays
  silent rather than treated as agreement or disagreement).
  `ensemble_probability_distance`/`ensemble_confidence_spread` are
  continuous and have no calibrated cutoff anywhere in this project
  (`docs/cv6_uncertainty_spec.md`), so no flag is invented for them --
  they remain on `CandidateResult.to_dict()` for a consumer who wants
  the raw numbers, without this module fabricating a threshold to
  summarize them into the JSON contract's `quality_flags`.

## The one real design decision: how CV-7 affects risk_category

A **one-way escalation ratchet**, never a de-escalation:

    GROWING or CHANGED_COLOR, with magnitude >= 1.0 (the delta actually
    crossed its calibrated threshold, not just a nonzero reading) AND
    confidence >= 2/3 (border+color both measured -- not a mostly-
    NO_PRIOR_DATA result) => risk_category moves up one step
    (LOW->MEDIUM->HIGH; HIGH stays HIGH), and requires_review becomes
    True regardless of what CV-4/CV-6 decided.

Why one-way and why these gates:

- **SHRINKING is never treated as reassuring**, and neither is STABLE
  or NO_PRIOR_DATA. "No visible growth" is equally consistent with
  "the lesion isn't changing" and "we couldn't measure it" --
  calibration's 4% size-coverage and measurement's 5% mask-miss rate
  are exactly this kind of silent gap (docs/cv7_temporal_technical_spec.md).
  A signal that can be silently absent must never be allowed to lower
  risk; it can only ever raise it when present and confident.
- **The magnitude >= 1.0 gate** matters because `LesionDelta.verdict`
  is already thresholded (delta.py only returns GROWING/CHANGED_COLOR
  once a feature crosses its calibrated threshold), so this looks
  redundant -- it is kept as an explicit, defensive re-check in case
  `magnitude` and `verdict` are ever computed independently upstream.
- **The confidence >= 2/3 gate** excludes escalating on a verdict that
  came from only 1 of the 3 feature channels being available (i.e., a
  border-only or color-only computation when the other channel and
  size are both missing) -- escalation should not ride on the
  thinnest possible evidence.
- **Confirmed statistically, not just designed this way**: Stage 1's
  own evaluation (analysis/quality/cv7_temporal_data/stage1_evaluation_result.md)
  found malignant-outcome lesions are ~2x as likely to register a
  non-STABLE verdict (p=0.0135) with a significantly higher magnitude
  distribution (p=8.2e-8) -- this escalation rule is acting on a signal
  already shown to correlate with the outcome it's meant to catch, not
  an untested assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.risk.action_mapping import ProductAction
from src.temporal.delta import TemporalVerdict
from src.temporal.pipeline import TemporalResult

if TYPE_CHECKING:
    # Import deferred to type-checking only: src.inference.orchestrator
    # imports assess_risk/RiskAssessment from this module, so a runtime
    # import here would be circular.
    from src.inference.orchestrator import CandidateResult


class RiskCategory(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


_BASE_RISK_CATEGORY: dict[ProductAction, RiskCategory] = {
    ProductAction.URGENT_EVALUATION: RiskCategory.HIGH,
    ProductAction.EVALUATE_SOON: RiskCategory.MEDIUM,
    ProductAction.MONITOR: RiskCategory.LOW,
    ProductAction.UNKNOWN: RiskCategory.HIGH,
}

_ESCALATE_ONE_STEP: dict[RiskCategory, RiskCategory] = {
    RiskCategory.LOW: RiskCategory.MEDIUM,
    RiskCategory.MEDIUM: RiskCategory.HIGH,
    RiskCategory.HIGH: RiskCategory.HIGH,
}

# Verdicts that can ever escalate risk. SHRINKING/STABLE/NO_PRIOR_DATA
# are deliberately excluded -- see module docstring.
_ESCALATING_VERDICTS = frozenset({TemporalVerdict.GROWING, TemporalVerdict.CHANGED_COLOR})

# Requires >=2 of 3 feature channels (border+color, since size needs
# calibration confident on both visits and is rare -- 0.3% per
# delta_calibration_result.md). See module docstring.
MIN_TEMPORAL_CONFIDENCE_FOR_ESCALATION = 2.0 / 3.0

# See module docstring's "quality_flags" section for provenance of
# both thresholds -- LOW_CROP_CONTRAST is independently validated at
# crop scale (docs/cv4_domain_evidence_spec.md); LOW_CROP_BLUR is
# borrowed from CV-1's whole-image advisory threshold, weaker evidence.
LOW_CROP_CONTRAST_THRESHOLD = 0.20
LOW_CROP_BLUR_THRESHOLD = 0.15

# Shape used when there is no second image to compare at all (a first
# upload, no visit history) -- semantically identical to LesionDelta's
# own NO_PRIOR_DATA case (compute_delta returns this same shape when a
# real pair fails), so both null-causes collapse into the one
# NO_PRIOR_DATA contract value. The locked contract has no separate
# slot for "history exists but was unmeasurable" vs. "no history at
# all"; this is a deliberate simplification, not an oversight.
_NO_COMPARISON_TEMPORAL: dict[str, Any] = {
    "verdict": TemporalVerdict.NO_PRIOR_DATA.value,
    "magnitude": 0.0,
    "confidence": 0.0,
    "per_feature_deltas": {"size": None, "border": None, "color": None},
    "compared_timestamps": [None, None],
}


@dataclass(frozen=True)
class RiskAssessment:
    """CV-8's convergent output. `to_dict()` is the locked JSON contract."""

    lesion_id: str
    native_class: str
    probabilities: dict[str, float]
    risk_category: RiskCategory
    risk_reason: str
    temporal: dict[str, Any]
    uncertainty_confidence: float
    requires_review: bool
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesion_id": self.lesion_id,
            "diagnosis": {
                "native_class": self.native_class,
                "probabilities": self.probabilities,
            },
            "risk_category": self.risk_category.value,
            "risk_reason": self.risk_reason,
            "temporal": self.temporal,
            "uncertainty": {
                "confidence": self.uncertainty_confidence,
                "requires_review": self.requires_review,
            },
            "quality_flags": list(self.quality_flags),
        }


def _temporal_escalates(temporal: TemporalResult | None) -> bool:
    if temporal is None:
        return False
    if temporal.verdict not in _ESCALATING_VERDICTS:
        return False
    if temporal.magnitude < 1.0:
        return False
    if temporal.confidence < MIN_TEMPORAL_CONFIDENCE_FOR_ESCALATION:
        return False
    return True


def _evidence_quality_flags(candidate: CandidateResult) -> list[str]:
    """
    Surface CV-1/CV-3/CV-6 evidence already on `candidate` as
    disclosure flags. Never affects risk_category or requires_review --
    see module docstring's "quality_flags" section for provenance of
    each threshold used here.
    """
    flags: list[str] = []
    if candidate.mask_degenerate:
        flags.append("DEGENERATE_MASK")
    if candidate.mask_touches_border:
        flags.append("MASK_TOUCHES_BORDER")
    if candidate.crop_contrast < LOW_CROP_CONTRAST_THRESHOLD:
        flags.append("LOW_CROP_CONTRAST")
    if candidate.crop_blur < LOW_CROP_BLUR_THRESHOLD:
        flags.append("LOW_CROP_BLUR")
    if candidate.ensemble_agree is False:
        flags.append("ENSEMBLE_DISAGREEMENT")
    return flags


def assess_risk(
    candidate: CandidateResult,
    *,
    lesion_id: str,
    temporal: TemporalResult | None = None,
    extra_quality_flags: tuple[str, ...] = (),
) -> RiskAssessment:
    """
    Converge CV-4 (via `candidate`) and CV-7 (via `temporal`) into one
    risk assessment. `temporal=None` means no prior-visit image was
    available to compare against at all -- or, per the orchestrator's
    own ambiguity rule, one was available but not applied (see
    `extra_quality_flags`, e.g. `PRIOR_IMAGE_PAIRING_AMBIGUOUS` from
    `src/inference/orchestrator.py::_resolve_temporal_pairing`).
    `extra_quality_flags` lets a caller record why, without this
    module needing to know about orchestrator-level concerns like
    multi-candidate ambiguity.
    """
    base_category = _BASE_RISK_CATEGORY[candidate.product_action]

    quality_flags: list[str] = list(extra_quality_flags) + _evidence_quality_flags(candidate)
    if temporal is None:
        temporal_dict = dict(_NO_COMPARISON_TEMPORAL)
        quality_flags.append("NO_TEMPORAL_COMPARISON")
    else:
        temporal_dict = temporal.to_dict()
        if temporal.verdict is TemporalVerdict.NO_PRIOR_DATA:
            quality_flags.append("TEMPORAL_NO_PRIOR_DATA")
        elif (
            temporal.verdict in _ESCALATING_VERDICTS
            and temporal.confidence < MIN_TEMPORAL_CONFIDENCE_FOR_ESCALATION
        ):
            quality_flags.append("TEMPORAL_LOW_CONFIDENCE")

    escalated = _temporal_escalates(temporal)
    risk_category = _ESCALATE_ONE_STEP[base_category] if escalated else base_category
    requires_review = candidate.requires_review or escalated

    reason = (
        f"{candidate.predicted_class} -> {candidate.product_action.value} "
        f"({candidate.confidence:.0%} confidence)"
    )
    if escalated:
        reason += (
            f"; escalated to {risk_category.value} due to "
            f"{temporal.verdict.value} (magnitude {temporal.magnitude:.2f})"
        )

    return RiskAssessment(
        lesion_id=lesion_id,
        native_class=candidate.predicted_class,
        probabilities=candidate.probabilities,
        risk_category=risk_category,
        risk_reason=reason,
        temporal=temporal_dict,
        uncertainty_confidence=candidate.calibrated_confidence,
        requires_review=requires_review,
        quality_flags=tuple(quality_flags),
    )
