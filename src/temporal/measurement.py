"""
CV-7 per-visit lesion measurement.

Reuses CV-3 directly for segmentation (src/segmentation/inference.py) --
no new segmentation model. Validated before building on top of it
(analysis/quality/cv7_temporal_data/measurement_result.md): only 5%
degenerate-empty masks on a 100-image random sample, far better than
CV-3's ~22% failure rate on iToBoS TBP crops, confirming these
dermoscopic close-ups match CV-3's actual training domain.

Two independent gates, matching calibration.py's fail-loud pattern:

  - `valid`: was a lesion found at all (mask non-empty)? If not,
    nothing here is measurable -- no size, color, or border.
  - `diameter_mm` / `area_mm2` being None (even when `valid`): was
    ruler calibration confident for this image? Border (`compactness`)
    and pixel-space `area_fraction` need no calibration and are always
    available when `valid` -- only real-unit size needs it. A failed
    calibration must never silently produce a wrong mm value.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.temporal.calibration import RulerCalibration


@dataclass(frozen=True)
class LesionMeasurement:
    """One visit's measurement of one lesion."""

    valid: bool
    reason: str

    # Scale-invariant, always available when valid=True.
    area_fraction: float | None
    compactness: float | None
    mean_lab: tuple[float, float, float] | None

    # Real-unit, only available when valid=True AND calibration was confident.
    diameter_mm: float | None
    area_mm2: float | None


def _largest_component_mask(mask: np.ndarray) -> np.ndarray | None:
    """
    Isolate the single largest connected foreground region.

    A mask can contain more than one blob (e.g. a lesion plus a nearby
    freckle, seen during validation) -- this module measures ONE
    lesion per image, matching the dataset's one-lesion-per-file
    naming convention, so only the largest component is kept.
    """
    mask_u8 = (mask > 0.5).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return (labels == largest_label).astype(np.uint8)


def measure_lesion(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    calibration: RulerCalibration | None,
) -> LesionMeasurement:
    """
    Measure one visit's lesion from CV-3's mask and (optionally) a
    confident ruler calibration.

    Args:
        image_bgr: the same image CV-3 was run on, at its ORIGINAL
            resolution (not CV-3's 512x512 input size) -- calibration's
            px/mm was measured in this image's own coordinates.
        mask: CV-3's raw output mask (typically 512x512). Resized here
            to image_bgr's resolution before any measurement, so pixel
            counts and calibration agree on the same coordinate space.
        calibration: result of `calibrate(image_bgr)`, or None if
            calibration was never attempted for this image.
    """
    height, width = image_bgr.shape[:2]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(
            mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST
        )

    component = _largest_component_mask(mask)
    if component is None:
        return LesionMeasurement(
            valid=False, reason="empty mask -- no lesion found",
            area_fraction=None, compactness=None, mean_lab=None,
            diameter_mm=None, area_mm2=None,
        )

    contours, _ = cv2.findContours(
        component * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return LesionMeasurement(
            valid=False, reason="mask had foreground but no extractable contour",
            area_fraction=None, compactness=None, mean_lab=None,
            diameter_mm=None, area_mm2=None,
        )

    contour = max(contours, key=cv2.contourArea)
    area_px = float(cv2.contourArea(contour))
    perimeter_px = float(cv2.arcLength(contour, closed=True))

    if area_px <= 0:
        return LesionMeasurement(
            valid=False, reason="degenerate contour (zero area)",
            area_fraction=None, compactness=None, mean_lab=None,
            diameter_mm=None, area_mm2=None,
        )

    area_fraction = area_px / (height * width)
    # Standard border-irregularity measure: 1.0 for a perfect circle,
    # higher for irregular/jagged borders. Scale-invariant by
    # construction -- no calibration needed.
    compactness = (perimeter_px ** 2) / (4 * np.pi * area_px)

    lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    mean_lab_values = cv2.mean(lab_image, mask=component * 255)[:3]
    mean_lab = (float(mean_lab_values[0]), float(mean_lab_values[1]), float(mean_lab_values[2]))

    diameter_mm = None
    area_mm2 = None
    if calibration is not None and calibration.confident:
        mm_per_pixel = calibration.mm_per_pixel()
        (_, _), radius_px = cv2.minEnclosingCircle(contour)
        diameter_mm = 2.0 * radius_px * mm_per_pixel
        area_mm2 = area_px * (mm_per_pixel ** 2)

    return LesionMeasurement(
        valid=True,
        reason="ok",
        area_fraction=area_fraction,
        compactness=compactness,
        mean_lab=mean_lab,
        diameter_mm=diameter_mm,
        area_mm2=area_mm2,
    )
