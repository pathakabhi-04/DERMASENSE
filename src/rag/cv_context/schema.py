"""
The RAG side's internal representation of one CV-8 risk assessment.

Kept deliberately separate from `RetrievedMedicalEvidence` and (later)
`PatientContext`, and combined only at prompt assembly -- primary spec
section 19. That separation is what makes the constrained-paraphrase
guarantee enforceable: CV context is evidence the LLM may explain, on
exactly the same footing as a retrieved medical passage, and never a
thing it may originate or override.

## Why rendering lives here and not in the prompt builder

`format_for_prompt()` is the ONLY place a CV assessment becomes text.
The prompt builder, the safety layer, and the fallback all call it, so
the rules about which numbers are safe to show are stated once instead
of three times.

Those rules come from the CV side's own contract notes
(`docs/cv8_contract_delta_v1.1.md`, `src/risk/convergence.py`):

- **Confidence is `uncertainty.confidence`, never the raw softmax.**
  CV-6 calibrates the raw score; the calibrated number is the only one
  meaningful as "how confident is this assessment". The two differ by
  1.8-7.7 points across the five real fixtures.
- **`magnitude` is never rendered numerically.** It is a unitless ratio
  against each feature's own escalation threshold (1.0 == exactly at
  threshold), so as a bare number it reads as a physical quantity.
- **`per_feature_deltas` values are never rendered numerically.** They
  are mixed-unit (size in mm, border a compactness difference, color a
  CIE Lab delta-E) and only `magnitude` is threshold-normalized. The
  thresholds are not part of the contract, so these numbers cannot be
  interpreted from the payload alone -- a real STABLE fixture carries
  `color: 20.5` against a 24.0 threshold.
- **`compared_timestamps` is never rendered as a date.** It is opaque
  caller-supplied passthrough that CV-7 never parses.
- **`null` is not `0.0`.** A null delta means "could not be measured";
  0.0 means "measured, no change". That distinction must survive into
  the text, because collapsing it is how "we couldn't measure your
  lesion" becomes "your lesion is unchanged".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Verdicts CV-7 can emit. The parser rejects anything else rather than
# narrating a verdict it does not understand.
TEMPORAL_VERDICTS = frozenset(
    {"STABLE", "GROWING", "SHRINKING", "CHANGED_COLOR", "NO_PRIOR_DATA"}
)

RISK_CATEGORIES = frozenset({"LOW", "MEDIUM", "HIGH"})

# Distinguishes the two NO_PRIOR_DATA causes (primary spec section 17).
# `verdict` alone cannot tell them apart -- both produce an identical
# temporal block -- so the flags are the only signal.
FLAG_NO_PRIOR_IMAGE_AT_ALL = "NO_TEMPORAL_COMPARISON"
FLAG_COMPARISON_ATTEMPTED_AND_FAILED = "TEMPORAL_NO_PRIOR_DATA"

# Human-readable disclosure text for the quality flags CV-8 emits today.
# Unknown flags are NOT an error (spec section 16): they are surfaced by
# their raw name, so a new CV-side flag degrades to a slightly clumsy
# phrase rather than a crash or a silent drop.
_FLAG_DESCRIPTIONS: dict[str, str] = {
    "DEGENERATE_MASK": "the lesion outline could not be established in this photo",
    "MASK_TOUCHES_BORDER": "the lesion touches the edge of the photo, so part of it may be cut off",
    "LOW_CROP_CONTRAST": "the photo has low contrast around the lesion",
    "LOW_CROP_BLUR": "the photo is somewhat blurred around the lesion",
    "ENSEMBLE_DISAGREEMENT": "the classification models disagreed with each other",
    "NO_TEMPORAL_COMPARISON": "no previous photo was available to compare against",
    "TEMPORAL_NO_PRIOR_DATA": "a previous photo was supplied, but the comparison could not be completed",
    "TEMPORAL_LOW_CONFIDENCE": "the comparison with the previous photo rested on limited measurements",
    "PRIOR_IMAGE_PAIRING_AMBIGUOUS": (
        "several lesions were detected alongside a previous photo, so it was unclear "
        "which one to compare"
    ),
}


@dataclass(frozen=True)
class TemporalContext:
    """
    CV-7's change assessment for one lesion.

    `magnitude` and `per_feature_deltas` are retained in full for audit
    and for any future consumer, but neither is rendered numerically --
    see the module docstring.
    """

    verdict: str
    magnitude: float
    confidence: float
    size_delta: float | None
    border_delta: float | None
    color_delta: float | None
    compared_images: tuple[str | None, str | None]

    @property
    def comparison_happened(self) -> bool:
        return self.verdict != "NO_PRIOR_DATA"

    @property
    def measured_features(self) -> list[str]:
        """
        Which of the three feature channels produced a real measurement.
        A `None` here means "could not be measured", never "no change".
        """

        return [
            name
            for name, value in (
                ("size", self.size_delta),
                ("border", self.border_delta),
                ("color", self.color_delta),
            )
            if value is not None
        ]


@dataclass(frozen=True)
class CVAssessmentContext:
    """
    One CV-8 `RiskAssessment` payload, parsed and validated.

    Construct via `src.rag.cv_context.parser.parse_cv_assessment`; this
    type does no validation of its own so that every fail-loud decision
    lives in one place.
    """

    lesion_id: str
    native_class: str
    probabilities: dict[str, float]
    risk_category: str
    risk_reason: str
    temporal: TemporalContext
    confidence: float
    requires_review: bool
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    contract_version: str | None = None

    # ---- rendering -----------------------------------------------------

    def _describe_change(self) -> str:
        t = self.temporal

        if t.verdict == "NO_PRIOR_DATA":
            # Spec section 17: the two causes are indistinguishable from
            # `verdict` alone and must be separated by the flags.
            if FLAG_COMPARISON_ATTEMPTED_AND_FAILED in self.quality_flags:
                return (
                    "a previous photo was supplied, but the comparison could not be "
                    "completed, so no change assessment is available"
                )
            if FLAG_NO_PRIOR_IMAGE_AT_ALL in self.quality_flags:
                return (
                    "no previous photo was available, so no change assessment was "
                    "performed"
                )
            return "no change assessment is available"

        described = {
            "STABLE": "no meaningful change was detected since the previous photo",
            "GROWING": "the lesion has grown beyond the threshold used to flag a meaningful change",
            "SHRINKING": "the lesion has shrunk beyond the threshold used to flag a meaningful change",
            "CHANGED_COLOR": (
                "the lesion's colour changed beyond the threshold used to flag a "
                "meaningful change"
            ),
        }[t.verdict]

        # Preserve null-vs-zero. Saying "stable" without this caveat is
        # how "we could not measure it" becomes "it did not change".
        unmeasured = [
            name
            for name, value in (
                ("size", t.size_delta),
                ("border shape", t.border_delta),
                ("colour", t.color_delta),
            )
            if value is None
        ]
        if unmeasured:
            described += (
                f" (note: {', '.join(unmeasured)} could not be measured in this "
                "comparison, which is not the same as being unchanged)"
            )

        return described

    def _describe_quality_flags(self) -> str:
        if not self.quality_flags:
            return "none"

        return "; ".join(
            _FLAG_DESCRIPTIONS.get(flag, f"an unrecognised quality flag ({flag})")
            for flag in self.quality_flags
        )

    def format_for_prompt(self, *, include_diagnosis: bool = True) -> str:
        """
        Render this assessment as supplied evidence for the LLM.

        Every figure here is safe to narrate; anything unsafe to narrate
        is deliberately absent rather than present-but-discouraged.

        `include_diagnosis=False` is **Plan A / narrow-product mode**
        (docs/plan_a_narrow_product_spec.md §5). It omits the class, the
        confidence attached to it, and the risk category, because the
        narrow product makes no diagnostic claim -- a user told "most
        likely class: NEV" has been reassured whether or not a risk
        category accompanied it.

        What survives is what the narrow product actually ships: the
        change-over-time verdict and the capture-quality notes. The
        default stays True so Phase 1/2 behaviour, and every existing
        test of it, is unchanged.
        """

        lines = [
            "CV ASSESSMENT (computed by the DermaSense image pipeline for this "
            "patient's own photo — explain it, never recompute or override it):",
            f"- Lesion identifier: {self.lesion_id}",
        ]
        if include_diagnosis:
            lines += [
                f"- Most likely class from the image classifier: {self.native_class}",
                f"- Assessment confidence: {self.confidence:.0%} (calibrated)",
                f"- Risk category: {self.risk_category}",
            ]
        lines += [
            f"- Flagged for professional review: {'yes' if self.requires_review else 'no'}",
            f"- Change since the previous photo: {self._describe_change()}",
            f"- Image quality notes: {self._describe_quality_flags()}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable view, for logging and tests."""

        return {
            "contract_version": self.contract_version,
            "lesion_id": self.lesion_id,
            "native_class": self.native_class,
            "probabilities": dict(self.probabilities),
            "risk_category": self.risk_category,
            "risk_reason": self.risk_reason,
            "temporal": {
                "verdict": self.temporal.verdict,
                "magnitude": self.temporal.magnitude,
                "confidence": self.temporal.confidence,
                "per_feature_deltas": {
                    "size": self.temporal.size_delta,
                    "border": self.temporal.border_delta,
                    "color": self.temporal.color_delta,
                },
                "compared_timestamps": list(self.temporal.compared_images),
            },
            "uncertainty": {
                "confidence": self.confidence,
                "requires_review": self.requires_review,
            },
            "quality_flags": list(self.quality_flags),
        }


def order_by_severity(
    contexts: list[CVAssessmentContext],
) -> list[CVAssessmentContext]:
    """
    Order lesions most-severe first (spec section 20).

    CV-8 emits one assessment per detected lesion, so a single user turn
    can carry several. The baseline rule is to lead with the highest
    `risk_category` and never silently drop the others; `requires_review`
    breaks ties, then calibrated confidence, so the ordering is total and
    deterministic rather than dependent on input order.
    """

    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    return sorted(
        contexts,
        key=lambda c: (rank[c.risk_category], not c.requires_review, -c.confidence),
    )


def render_cv_context(
    cv_context: "CVAssessmentContext | list[CVAssessmentContext] | None",
    *,
    include_diagnosis: bool = True,
) -> str:
    """
    Render zero, one, or several CV assessments to text.

    Lives here, beside the schema, because three separate consumers need
    it -- the prompt builder, the grounding check, and the fallback --
    and they must agree EXACTLY on what the CV evidence says. If the
    grounding check scored an answer against different text than the
    prompt supplied or the fallback displayed, "grounded" would stop
    meaning anything.

    Several lesions are ordered most-severe-first (spec section 20) and
    none is dropped: CV-8 emits one assessment per detected lesion, so a
    single turn genuinely can carry more than one.
    """

    if cv_context is None:
        return ""

    contexts = (
        list(cv_context) if isinstance(cv_context, (list, tuple)) else [cv_context]
    )

    if not contexts:
        return ""

    if len(contexts) > 1:
        contexts = order_by_severity(contexts)

    return "\n\n".join(
        context.format_for_prompt(include_diagnosis=include_diagnosis)
        for context in contexts
    )
