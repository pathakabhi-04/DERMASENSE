"""
DermaSense CV-1.5 domain router -- Stage 1 (classical heuristic).

Product contract:

    raw image
        |
    pigmentation-contrast blob analysis
        |
    pre_framed / wide_field

This module does not perform lesion diagnosis or detection. It answers
one question: does this image look like a close-up, lesion-centric
clinical photo (route straight to CV-3) or a wide-field body-region
photo (route through CV-2 first)? See docs/cv1_5_router_spec.md for the
pre-committed evaluation and escalation criteria.

Signal: a pre-framed clinical photo is typically dominated by one large,
distinctly-pigmented lesion region filling a substantial fraction of the
frame. A wide-field photo shows a much larger area of relatively uniform
skin, with lesions (if visible at all) occupying only a small fraction
of the frame, often as several small, scattered regions.

Threshold below is calibrated once against a train-split calibration
sample (200 images, not the held-out evaluation set in the spec) via a
one-time sweep -- see docs/cv1_5_router_spec.md. Calibration finding:
blob COUNT is not a useful discriminator in the hypothesized direction
-- pre_framed close-ups actually show MORE small pigmented regions than
wide_field photos (closer zoom resolves more visible texture: freckles,
hair, marker dots), not fewer. Route decision uses largest-blob-fraction
alone; the significant_blob_count field is retained on FramingSignal for
diagnostics but does not gate the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

FramingLabel = Literal["pre_framed", "wide_field"]


@dataclass(frozen=True)
class FramingSignal:
    """Diagnostic signal underlying the pre_framed / wide_field decision."""

    largest_blob_fraction: float
    significant_blob_count: int
    label: FramingLabel


def _validate_image(image_bgr: np.ndarray) -> np.ndarray:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy.ndarray")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape [H,W,3]")
    return image_bgr


def _pigmentation_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Flag pixels that are substantially darker/more saturated than the
    image's dominant (assumed-skin) tone, robust to per-image lighting.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)

    # Dominant skin tone estimated per-image (median is robust to a
    # lesion occupying a minority of the frame, which holds even for
    # pre_framed images since the lesion is large but not the majority
    # of pixels in nearly all clinical photos).
    median_value = float(np.median(value))
    median_saturation = float(np.median(saturation))

    # A pixel is "pigmented" if it is darker AND more saturated than the
    # image's own dominant tone -- catches lesions across lighting
    # conditions without a fixed absolute color threshold.
    darker = value < (median_value - 20.0)
    more_saturated = saturation > (median_saturation + 10.0)

    mask = (darker & more_saturated).astype(np.uint8) * 255

    # Light morphological cleanup -- remove salt-noise, close small gaps
    # within a single lesion region.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def framing_signal(
    image_bgr: np.ndarray,
    *,
    min_blob_area_frac: float = 0.001,
) -> FramingSignal:
    """
    Compute the raw framing signal without applying the decision
    thresholds (kept separate so calibration can sweep decision
    thresholds without recomputing the blob analysis).
    """
    image_bgr = _validate_image(image_bgr)
    height, width = image_bgr.shape[:2]
    frame_area = float(height * width)

    mask = _pigmentation_mask(image_bgr)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    min_area_px = min_blob_area_frac * frame_area

    # label 0 is background
    areas = [
        stats[i, cv2.CC_STAT_AREA]
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px
    ]

    largest_fraction = (max(areas) / frame_area) if areas else 0.0
    blob_count = len(areas)

    return FramingSignal(
        largest_blob_fraction=largest_fraction,
        significant_blob_count=blob_count,
        label="pre_framed",  # placeholder, set by route_image
    )


def route_image(
    image_bgr: np.ndarray,
    *,
    blob_fraction_threshold: float = 0.005,
) -> FramingLabel:
    """
    Decide whether an image is pre_framed (lesion-centric close-up) or
    wide_field (body-region photo needing CV-2 detection first).

    Args:
        image_bgr: source image as loaded by cv2.imread (BGR, HxWx3 uint8).
        blob_fraction_threshold: minimum fraction of the frame the
            largest pigmentation blob must occupy to call the image
            pre_framed. Calibrated (train-split sweep, see module
            docstring) -- best balanced accuracy on the calibration
            sample was ~0.75, well short of the spec's 0.90 per-class
            gate. This is a low, near-"any detectable blob" threshold;
            it is the calibration optimum, not a robust separator.
    """
    signal = framing_signal(image_bgr)
    is_framed = signal.largest_blob_fraction >= blob_fraction_threshold
    return "pre_framed" if is_framed else "wide_field"
