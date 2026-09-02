"""
CV-7 pairwise delta computation and verdict assignment.

Consumes two `LesionMeasurement`s (src/temporal/measurement.py) for the
same lesion at different visits and produces the structured verdict
locked in docs/cv7_temporal_rag_integration_spec.md's JSON contract:
`STABLE | GROWING | SHRINKING | CHANGED_COLOR | NO_PRIOR_DATA`, plus
per-feature deltas for size/border/color and a confidence.

Two independent availability gates, same fail-loud pattern as
calibration.py and measurement.py:

  - Size delta (`size_delta_mm`) requires BOTH visits to have a
    confident ruler calibration. Pixel-space area/diameter is
    deliberately never compared across visits even when both masks are
    valid: these are handheld photos, so a lesion's fraction of the
    frame changes with camera distance, not just real growth. Without
    a real-unit (mm) scale on both sides, "size changed" cannot be
    told apart from "the photo was taken closer." This is the reason
    size-change detection inherits calibration's 4.0% coverage
    (analysis/quality/cv7_temporal_data/calibration_result.md).
  - Border (`border_delta`, from compactness) and color (`color_delta`,
    CIE76 Lab distance) need no calibration -- both are computed from
    ratios/differences that don't depend on absolute scale -- so they
    are available whenever both visits have a valid mask (~90% of
    pairs, given measurement.py's ~5% per-image miss rate).

The locked contract has no "CHANGED_BORDER" verdict -- border
irregularity is always recorded as evidence (`per_feature_deltas.border`)
but, by the contract's own design, never sets the headline verdict on
its own. Only size (GROWING/SHRINKING) and color (CHANGED_COLOR) do,
in that priority order when both cross their threshold at once, since
growth is the more directly actionable ABCDE signal.

Border and color thresholds below are calibrated against real deltas
measured on a 300-pair sample of the staged data -- see
analysis/quality/cv7_temporal_data/delta_calibration_result.md and
scripts/calibrate_cv7_thresholds.py. The size (growth) threshold is
NOT calibrated the same way: the sample yielded only 1 pair (0.3%)
with confident calibration on BOTH visits -- the direct, compounding
consequence of calibration's own 4.0% single-image coverage. There is
not enough real double-confident data yet to set this threshold
empirically; it is a provisional placeholder pending either more
double-confident data or an independently-corroborated clinical figure
(the same evidentiary bar used for the 1mm/tick assumption in
calibration.py) -- not attempted now, per this project's
anti-rabbit-hole discipline: chasing enough double-confident pairs to
calibrate this properly would mean processing most of the 8,751-image
staged corpus for a feature that will remain rare regardless (each
image independently has only ~4% odds of confident calibration).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from src.temporal.measurement import LesionMeasurement

# NOT calibrated from real data -- see module docstring. Provisional.
GROWTH_PCT_THRESHOLD = 20.0
GROWTH_ABS_MM_FLOOR = 0.5

# Calibrated against analysis/quality/cv7_temporal_data/delta_calibration_result.md
# (p90 of a real 300-pair sample, n=280 with both masks valid).
COMPACTNESS_DELTA_THRESHOLD = 3.0
COLOR_DELTA_E_THRESHOLD = 24.0


class TemporalVerdict(str, Enum):
    STABLE = "STABLE"
    GROWING = "GROWING"
    SHRINKING = "SHRINKING"
    CHANGED_COLOR = "CHANGED_COLOR"
    NO_PRIOR_DATA = "NO_PRIOR_DATA"


@dataclass(frozen=True)
class LesionDelta:
    verdict: TemporalVerdict
    magnitude: float
    confidence: float
    reason: str

    size_delta_mm: float | None
    size_pct_change: float | None
    border_delta: float | None
    color_delta: float | None


def _color_distance(lab_a: tuple[float, float, float], lab_b: tuple[float, float, float]) -> float:
    """CIE76 Lab distance -- a standard, simple perceptual color-difference metric."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lab_a, lab_b)))


def compute_delta(earlier: LesionMeasurement, later: LesionMeasurement) -> LesionDelta:
    """
    Compute the temporal delta between two visits of the same lesion.

    `earlier` and `later` must already be ordered by visit time --
    this function does not know which visit came first.
    """
    if not earlier.valid or not later.valid:
        return LesionDelta(
            verdict=TemporalVerdict.NO_PRIOR_DATA,
            magnitude=0.0,
            confidence=0.0,
            reason="no lesion mask found in at least one visit",
            size_delta_mm=None,
            size_pct_change=None,
            border_delta=None,
            color_delta=None,
        )

    size_delta_mm = None
    size_pct_change = None
    if earlier.diameter_mm is not None and later.diameter_mm is not None:
        size_delta_mm = later.diameter_mm - earlier.diameter_mm
        if earlier.diameter_mm > 0:
            size_pct_change = (size_delta_mm / earlier.diameter_mm) * 100.0

    border_delta = later.compactness - earlier.compactness
    color_delta = _color_distance(earlier.mean_lab, later.mean_lab)

    features_available = 1 + 1 + (1 if size_delta_mm is not None else 0)  # border, color, size
    confidence = features_available / 3.0

    size_ratio = 0.0
    if size_pct_change is not None and abs(size_delta_mm) >= GROWTH_ABS_MM_FLOOR:
        size_ratio = size_pct_change / GROWTH_PCT_THRESHOLD
    color_ratio = color_delta / COLOR_DELTA_E_THRESHOLD
    border_ratio = border_delta / COMPACTNESS_DELTA_THRESHOLD

    if size_ratio >= 1.0:
        verdict = TemporalVerdict.GROWING
        magnitude = size_ratio
        reason = f"diameter grew {size_pct_change:.1f}% ({size_delta_mm:+.2f}mm)"
    elif size_ratio <= -1.0:
        verdict = TemporalVerdict.SHRINKING
        magnitude = abs(size_ratio)
        reason = f"diameter shrank {size_pct_change:.1f}% ({size_delta_mm:+.2f}mm)"
    elif color_ratio >= 1.0:
        verdict = TemporalVerdict.CHANGED_COLOR
        magnitude = color_ratio
        reason = f"Lab color distance {color_delta:.1f} exceeded threshold"
    else:
        verdict = TemporalVerdict.STABLE
        magnitude = max(abs(size_ratio), color_ratio, abs(border_ratio))
        reason = "no feature delta crossed its calibrated threshold"

    return LesionDelta(
        verdict=verdict,
        magnitude=magnitude,
        confidence=confidence,
        reason=reason,
        size_delta_mm=size_delta_mm,
        size_pct_change=size_pct_change,
        border_delta=border_delta,
        color_delta=color_delta,
    )
