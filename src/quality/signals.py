"""
DermaSense CV-1 image-quality signals.

These signals measure objective properties of the input image.
They do not make diagnostic claims.

CV-1 v1 intentionally uses deterministic image statistics rather
than a learned quality model.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualitySignal:
    """Normalized quality signal."""

    name: str
    score: float
    value: float
    severity: float


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "image must be a numpy.ndarray"
        )

    if image.size == 0:
        raise ValueError(
            "image must not be empty"
        )

    if image.ndim not in (2, 3):
        raise ValueError(
            "image must have shape [H,W] or [H,W,C]"
        )

    return image


def _grayscale(image: np.ndarray) -> np.ndarray:
    image = _validate_image(image)

    if image.ndim == 2:
        return image.astype(np.uint8)

    if image.shape[2] == 1:
        return image[:, :, 0].astype(np.uint8)

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )


def resolution_signal(
    image: np.ndarray,
    *,
    minimum_dimension: int = 256,
) -> QualitySignal:
    """
    Assess effective spatial detail rather than only image dimensions.

    An image can have large pixel dimensions while still containing very
    little usable detail because of downsampling, compression, or other
    capture degradation. This signal therefore combines:

      - physical image dimensions
      - effective high-frequency detail

    The raw detail measurement is retained in ``value`` so that the
    operating range can be calibrated against real DermaSense images.
    """

    image = _validate_image(image)

    height, width = image.shape[:2]
    minimum = min(height, width)

    if minimum_dimension <= 0:
        raise ValueError(
            "minimum_dimension must be positive"
        )

    dimension_score = min(
        1.0,
        minimum / float(minimum_dimension),
    )

    gray = _grayscale(image)

    laplacian_variance = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    # Effective-detail reference derived from the current engineering
    # calibration experiments. This is intentionally not a clinical
    # threshold.
    detail_reference = 25.0

    detail_score = float(
        np.clip(
            laplacian_variance / detail_reference,
            0.0,
            1.0,
        )
    )

    score = float(
        min(
            dimension_score,
            detail_score,
        )
    )

    return QualitySignal(
        name="resolution",
        score=score,
        value=laplacian_variance,
        severity=1.0 - score,
    )

def brightness_signal(
    image: np.ndarray,
) -> QualitySignal:
    """
    Assess overall exposure.

    The score is highest around a moderate mean luminance and decreases
    toward severe underexposure or overexposure.
    """

    gray = _grayscale(image)

    mean_luminance = float(
        np.mean(gray) / 255.0
    )

    # Broad initial operating range.
    ideal = 0.50
    half_range = 0.45

    score = 1.0 - (
        abs(mean_luminance - ideal)
        / half_range
    )

    score = float(
        np.clip(score, 0.0, 1.0)
    )

    return QualitySignal(
        name="brightness",
        score=score,
        value=mean_luminance,
        severity=1.0 - score,
    )


def contrast_signal(
    image: np.ndarray,
) -> QualitySignal:
    """Assess global luminance contrast."""

    gray = _grayscale(image)

    contrast = float(
        np.std(gray) / 255.0
    )

    # Initial engineering reference range.
    target_contrast = 0.20

    score = contrast / target_contrast
    score = float(
        np.clip(score, 0.0, 1.0)
    )

    return QualitySignal(
        name="contrast",
        score=score,
        value=contrast,
        severity=1.0 - score,
    )


def blur_signal(
    image: np.ndarray,
) -> QualitySignal:
    """
    Estimate focus using variance of the Laplacian.

    Higher variance generally indicates stronger high-frequency detail.
    The raw value is retained so thresholds can later be calibrated on
    real DermaSense image distributions.
    """

    gray = _grayscale(image)

    laplacian_variance = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    # Conservative initial reference value.
    reference = 100.0

    score = laplacian_variance / reference
    score = float(
        np.clip(score, 0.0, 1.0)
    )

    return QualitySignal(
        name="blur",
        score=score,
        value=laplacian_variance,
        severity=1.0 - score,
    )
