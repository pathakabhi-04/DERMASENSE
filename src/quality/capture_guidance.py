"""
DermaSense capture guidance.

Turns CV-1 quality signals and the CV-1.5 framing decision into an
ordered list of concrete suggestions the product can show a user who
needs to recapture.

Why this is a safety mechanism and not UX polish
(docs/cv1_cv4_assembly_spec.md, decision 4): steering a user toward a
close-up routes them onto the only branch with validated end-to-end
evidence. The pre-framed branch runs CV-4 on the distribution it was
validated against; the wide-field branch stacks CV-2's ~19% silent-miss
rate, CV-3's ~22% TBP mask fragmentation, and CV-4 classifying crops
from a domain it was never validated on.

Why it warns rather than blocks: a user cannot photograph a mole on
their own back or shoulder -- sites where melanoma is most commonly
missed in men. Hard-blocking a wide-field submission would
systematically exclude the highest-risk anatomy from the product. So
this module suggests; it never decides to reject.

This module returns STRUCTURED suggestions. User-facing copy tone,
ordering in the UI, and whether to offer "proceed anyway" are product
decisions made downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.quality.assessment import QualityResult
from src.quality.guidance import guidance_for_issue

FRAMING_SUGGESTION = (
    "Move the camera closer so the mole or lesion fills most of the "
    "frame, then retake the image."
)

WIDE_FIELD_CAVEAT = (
    "This looks like a wide photo of a body area rather than a close-up "
    "of a single lesion. It will be screened for possible lesions, but a "
    "close-up gives a more reliable assessment."
)


@dataclass(frozen=True)
class CaptureSuggestion:
    """One actionable suggestion for recapturing an image."""

    category: str  # "quality" | "framing"
    issue: str
    guidance: str
    severity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "issue": self.issue,
            "guidance": self.guidance,
            "severity": self.severity,
        }


def build_capture_suggestions(
    quality: QualityResult,
    framing: str | None = None,
) -> tuple[CaptureSuggestion, ...]:
    """
    Build ordered capture suggestions.

    Quality issues come first, framing second -- guidance order follows
    pipeline order (CV-1 before CV-1.5). Telling someone to "move
    closer" on an image too blurry to interpret is not useful; fix
    interpretability first, framing second.

    Args:
        quality: the CV-1 result for this image.
        framing: the CV-1.5 routing decision ("pre_framed" or
            "wide_field"), or None if routing was not reached (e.g. the
            image was rejected on quality).

    Returns:
        Ordered suggestions, most severe quality issue first. Empty if
        the image is usable and already well framed.
    """
    suggestions: list[CaptureSuggestion] = []

    # CV-1 already sorts issues by descending severity.
    for issue in quality.issues:
        suggestions.append(
            CaptureSuggestion(
                category="quality",
                issue=issue.type,
                guidance=guidance_for_issue(issue.type),
                severity=issue.severity,
            )
        )

    if framing == "wide_field":
        suggestions.append(
            CaptureSuggestion(
                category="framing",
                issue="wide_field_framing",
                guidance=FRAMING_SUGGESTION,
                # Advisory, not disqualifying -- the wide-field branch
                # still processes the image.
                severity=0.5,
            )
        )

    return tuple(suggestions)
