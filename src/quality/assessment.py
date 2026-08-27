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
    minimum_quality_score: float = 0.50,
    resolution_threshold: float = 0.50,
    brightness_threshold: float = 0.40,
    contrast_threshold: float = 0.25,
    blur_threshold: float = 0.15,
) -> QualityResult:
    """
    Assess whether an image is suitable for downstream CV.

    Thresholds are initial engineering defaults. They are not clinical
    thresholds and must be validated against representative product
    images before deployment.
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

    usable = (
        quality_score >= minimum_quality_score
        and not issues
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
