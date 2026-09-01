"""
DermaSense CV-1 image quality assessment.

Product contract:

    raw image
        ↓
    objective quality signals
        ↓
    quality interpretation
        ↓
    actionable capture guidance

This module does not perform lesion diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.quality.guidance import guidance_for_issue
from src.quality.signals import (
    QualitySignal,
    blur_signal,
    brightness_signal,
    contrast_signal,
    resolution_signal,
)


@dataclass(frozen=True)
class QualityIssue:
    """A detected image-quality problem."""

    type: str
    severity: float
    guidance: str


@dataclass(frozen=True)
class QualityResult:
    """Public CV-1 result."""

    usable: bool
    quality_score: float
    issues: tuple[QualityIssue, ...]
    signals: dict[str, float]
    recommended_action: str

    @property
    def review_required(self) -> bool:
        """Whether the image should be stopped before downstream CV."""
        return not self.usable

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result for downstream product layers."""
        return {
            "usable": self.usable,
            "quality_score": self.quality_score,
            "issues": [
                {
                    "type": issue.type,
                    "severity": issue.severity,
                    "guidance": issue.guidance,
                }
                for issue in self.issues
            ],
            "signals": dict(self.signals),
            "recommended_action": self.recommended_action,
        }


def _issue(
    issue_type: str,
    severity: float,
) -> QualityIssue:
    return QualityIssue(
        type=issue_type,
        severity=float(np.clip(severity, 0.0, 1.0)),
        guidance=guidance_for_issue(issue_type),
    )


def assess_image(
    image: np.ndarray,
    *,
    minimum_dimension: int = 256,
    minimum_quality_score: float = 0.35,
    resolution_threshold: float = 0.50,
    brightness_threshold: float = 0.40,
    contrast_threshold: float = 0.25,
    blur_threshold: float = 0.15,
    unusable_resolution: float = 0.50,
    unusable_brightness: float = 0.30,
    unusable_contrast: float = 0.20,
    unusable_blur: float = 0.05,
) -> QualityResult:
    """
    Assess whether an image is suitable for downstream CV.

    Two tiers, deliberately separated (docs/cv1_recalibration_spec.md):

    - The ``*_threshold`` values flag ADVISORY issues. These populate
      ``issues`` so the capture-guidance layer can suggest a better
      retake, but on their own they do not block the image.
    - The ``unusable_*`` values are the BLOCKING tier. An image is
      rejected only when a signal falls into genuinely-unusable
      territory, or the composite score collapses.

    Previously any single advisory issue rejected the image
    (``usable = score >= 0.50 and not issues``). Measured against real
    data that gate rejected 13.6% of PAD-UFES clinical images -- images
    CV-4 then classified MORE accurately than the ones that passed
    (85.4% vs 67.8%), with no CV-1 signal predicting CV-4 success. The
    thresholds had been calibrated on synthetic degradation, a range
    real clinical images do not occupy, so the gate fired on normal
    variation instead of genuine unusability.

    Blocking thresholds are calibrated to catch severe synthetic
    degradation while admitting real clinical images -- see the spec for
    the two-sided acceptance criteria.
    """

    signals: list[QualitySignal] = [
        resolution_signal(
            image,
            minimum_dimension=minimum_dimension,
        ),
        brightness_signal(image),
        contrast_signal(image),
        blur_signal(image),
    ]

    signal_map = {
        signal.name: signal.score
        for signal in signals
    }

    quality_score = float(
        np.mean(
            [
                signal.score
                for signal in signals
            ]
        )
    )

    issues: list[QualityIssue] = []

    resolution = signal_map["resolution"]
    brightness = signal_map["brightness"]
    contrast = signal_map["contrast"]
    blur = signal_map["blur"]

    if resolution < resolution_threshold:
        issues.append(
            _issue(
                "resolution",
                1.0 - resolution,
            )
        )

    if brightness < brightness_threshold:
        raw_brightness = next(
            signal.value
            for signal in signals
            if signal.name == "brightness"
        )

        issue_type = (
            "low_brightness"
            if raw_brightness < 0.50
            else "high_brightness"
        )

        issues.append(
            _issue(
                issue_type,
                1.0 - brightness,
            )
        )

    if contrast < contrast_threshold:
        issues.append(
            _issue(
                "low_contrast",
                1.0 - contrast,
            )
        )

    if blur < blur_threshold:
        issues.append(
            _issue(
                "motion_blur",
                1.0 - blur,
            )
        )

    # Blocking tier: reject only on genuine unusability, not on any
    # advisory issue. Advisory issues still populate `issues` so the
    # capture-guidance layer can suggest a better retake.
    unusable_limits = {
        "resolution": unusable_resolution,
        "brightness": unusable_brightness,
        "contrast": unusable_contrast,
        "blur": unusable_blur,
    }

    blocking = [
        name
        for name, limit in unusable_limits.items()
        if signal_map[name] < limit
    ]

    usable = (
        quality_score >= minimum_quality_score
        and not blocking
    )

    recommended_action = (
        "PROCEED"
        if usable
        else "RETAKE"
    )

    issues.sort(
        key=lambda item: item.severity,
        reverse=True,
    )

    return QualityResult(
        usable=usable,
        quality_score=quality_score,
        issues=tuple(issues),
        signals=signal_map,
        recommended_action=recommended_action,
    )
