"""
CV-7 ruler calibration.

Every UQ Longitudinal dermoscopic image has a physical mm ruler etched
into the frame (bottom-left). This module detects its tick marks and
converts pixel distances to millimeters, per image -- never a fixed
global constant, since multiple cameras were used (Canon EOS Rebel T6i,
Veos SLR) with different fields of view (docs/cv7_temporal_technical_spec.md).

Tick interval assumption: 1mm per tick. This is not stated explicitly
in the dataset's own documentation (checked: the archive's
FurtherInformation.txt only points back to the eSpace listing page,
which does not cover it either) -- it is corroborated, not proven, by
two independent sources: Canfield's own spec sheet for the VEOS SLR
(one of the two cameras actually used in this study) states "270
pixels/mm" via its etched contact-plate scale, and this module's own
direct pixel measurement on a Canon-camera image independently landed
at ~266px between consecutive ticks -- within 1.5% of that figure on a
different camera. Because this is corroborated rather than certain,
EXPECTED_PX_PER_MM_RANGE below is used as a plausibility check that can
reject a bad detection, not as a value ever assumed without measuring.

Detection is imperfect (validated against 6 real images spanning both
cameras: 4 landed within the expected range, 1 gave an out-of-range
outlier, 1 found nothing) -- calibration failure is therefore a
first-class, expected outcome here, the same way QUALITY_REJECTED and
NO_CANDIDATES are first-class outcomes in the assembled CV-1->CV-4
pipeline (src/inference/orchestrator.py). A candidate whose calibration
fails should be treated as NO_PRIOR_DATA for the size dimension, never
silently measured with a guessed scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Corroborated by Canfield's VEOS SLR spec ("270 pixels/mm") and this
# module's own direct measurement (~266px/mm on a Canon-camera image).
# A plausibility band, not a hardcoded value -- calibration always
# measures per image; this only judges whether the measurement is
# believable.
EXPECTED_PX_PER_MM_RANGE = (220.0, 320.0)

MIN_TICKS_FOR_CONFIDENT_CALIBRATION = 4
MAX_SPACING_RELATIVE_STD = 0.08  # ticks must be evenly spaced, not just numerous


@dataclass(frozen=True)
class RulerCalibration:
    """Result of one calibration attempt on one image."""

    px_per_mm: float | None
    confident: bool
    num_ticks_detected: int
    reason: str

    def mm_per_pixel(self) -> float | None:
        if self.px_per_mm is None:
            return None
        return 1.0 / self.px_per_mm


def _detect_horizontal_ticks(
    image_bgr: np.ndarray,
) -> list[tuple[float, float, float, float]]:
    """
    Find near-horizontal line segments in the ruler region via a
    probabilistic Hough transform -- chosen over blob/connected-
    component heuristics because it directly targets "short straight
    segment at a known angle", which naturally rejects most hair
    (rarely perfectly straight and horizontal for the required length)
    far more robustly than darkness+shape thresholds alone (both were
    tried; Hough is what generalized across cameras in validation).

    Returns [(x_start, x_end, y_center, length), ...] in full-image
    coordinates, restricted to the region where the ruler sits (above
    the "mm" text label, left ~15% of the frame).
    """
    height, width = image_bgr.shape[:2]
    y0, y1 = int(height * 0.40), int(height * 0.90)
    x1 = int(width * 0.15)
    roi = image_bgr[y0:y1, 0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=40,
        minLineLength=50, maxLineGap=4,
    )
    if lines is None:
        return []

    horizontal = []
    for x_a, y_a, x_b, y_b in lines.reshape(-1, 4):
        angle = abs(np.degrees(np.arctan2(int(y_b) - int(y_a), int(x_b) - int(x_a))))
        if angle < 3 or angle > 177:
            length = float(np.hypot(x_b - x_a, y_b - y_a))
            y_center = (y_a + y_b) / 2.0 + y0
            horizontal.append(
                (float(min(x_a, x_b)), float(max(x_a, x_b)), y_center, length)
            )
    return horizontal


def _merge_nearby(
    segments: list[tuple[float, float, float, float]],
    y_tolerance: float = 8.0,
) -> list[tuple[float, float, float]]:
    """Merge multiple edge segments belonging to the same tick dash."""
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s[2])
    merged = [list(segments[0][:3])]
    for x_start, x_end, y_center, _ in segments[1:]:
        if y_center - merged[-1][2] < y_tolerance:
            merged[-1][0] = min(merged[-1][0], x_start)
            merged[-1][1] = max(merged[-1][1], x_end)
            merged[-1][2] = (merged[-1][2] + y_center) / 2.0
        else:
            merged.append([x_start, x_end, y_center])
    return [tuple(m) for m in merged]


def calibrate(image_bgr: np.ndarray) -> RulerCalibration:
    """
    Detect the ruler and compute a px/mm scale for this specific image.

    Never returns a guessed value: if too few ticks are found, if their
    spacing is inconsistent, or if the resulting scale falls outside
    the corroborated plausibility range, `confident` is False and
    `px_per_mm` is None -- callers must treat that candidate as
    NO_PRIOR_DATA for size, not silently measure it.
    """
    segments = _detect_horizontal_ticks(image_bgr)
    ticks = _merge_nearby(segments)

    if len(ticks) < 2:
        return RulerCalibration(
            px_per_mm=None, confident=False, num_ticks_detected=len(ticks),
            reason="fewer than 2 tick candidates detected",
        )

    y_centers = sorted(t[2] for t in ticks)
    diffs = np.diff(y_centers)
    plausible = diffs[(diffs > 100) & (diffs < 400)]

    if len(plausible) < MIN_TICKS_FOR_CONFIDENT_CALIBRATION - 1:
        return RulerCalibration(
            px_per_mm=None, confident=False, num_ticks_detected=len(ticks),
            reason=(
                f"only {len(plausible)} plausible tick-to-tick gaps "
                f"(need >= {MIN_TICKS_FOR_CONFIDENT_CALIBRATION - 1})"
            ),
        )

    spacing = float(np.median(plausible))
    relative_std = float(np.std(plausible) / spacing) if spacing > 0 else float("inf")

    if relative_std > MAX_SPACING_RELATIVE_STD:
        return RulerCalibration(
            px_per_mm=None, confident=False, num_ticks_detected=len(ticks),
            reason=f"tick spacing too irregular (relative std {relative_std:.2f})",
        )

    px_per_mm = spacing  # 1mm per tick, see module docstring
    low, high = EXPECTED_PX_PER_MM_RANGE
    if not (low <= px_per_mm <= high):
        return RulerCalibration(
            px_per_mm=None, confident=False, num_ticks_detected=len(ticks),
            reason=(
                f"measured {px_per_mm:.1f} px/mm outside plausible range "
                f"[{low}, {high}]"
            ),
        )

    return RulerCalibration(
        px_per_mm=px_per_mm,
        confident=True,
        num_ticks_detected=len(ticks),
        reason="ok",
    )
