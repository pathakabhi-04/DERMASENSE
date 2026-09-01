"""
Unit tests for CV-1/CV-1.5 capture guidance.

Pure function, no checkpoints required.
"""

from __future__ import annotations

import numpy as np

from src.quality.assessment import assess_image
from src.quality.capture_guidance import (
    FRAMING_SUGGESTION,
    build_capture_suggestions,
)


def _clean_image() -> np.ndarray:
    """A well-exposed, detailed image that should pass CV-1."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)


def _dark_blurry_image() -> np.ndarray:
    """A very dark, flat image that should fail CV-1."""
    return np.full((512, 512, 3), 5, dtype=np.uint8)


def test_no_suggestions_for_usable_pre_framed_image():
    quality = assess_image(_clean_image())

    suggestions = build_capture_suggestions(quality, "pre_framed")

    assert quality.usable
    assert suggestions == ()


def test_wide_field_adds_framing_suggestion():
    quality = assess_image(_clean_image())

    suggestions = build_capture_suggestions(quality, "wide_field")

    framing = [s for s in suggestions if s.category == "framing"]
    assert len(framing) == 1
    assert framing[0].issue == "wide_field_framing"
    assert framing[0].guidance == FRAMING_SUGGESTION


def test_quality_issues_precede_framing():
    """Guidance order follows pipeline order: fix quality, then framing."""
    quality = assess_image(_dark_blurry_image())

    suggestions = build_capture_suggestions(quality, "wide_field")

    assert not quality.usable
    categories = [s.category for s in suggestions]
    assert "quality" in categories
    assert categories.index("framing") == len(categories) - 1
    # every quality suggestion comes before the framing one
    assert all(
        category == "quality" for category in categories[:-1]
    )


def test_quality_suggestions_sorted_by_descending_severity():
    quality = assess_image(_dark_blurry_image())

    suggestions = build_capture_suggestions(quality, None)

    severities = [
        s.severity for s in suggestions if s.category == "quality"
    ]
    assert severities == sorted(severities, reverse=True)


def test_framing_omitted_when_routing_not_reached():
    """A quality-rejected image never reaches CV-1.5, so no framing hint."""
    quality = assess_image(_dark_blurry_image())

    suggestions = build_capture_suggestions(quality, None)

    assert all(s.category == "quality" for s in suggestions)


def test_suggestions_are_serializable():
    quality = assess_image(_dark_blurry_image())

    suggestions = build_capture_suggestions(quality, "wide_field")

    for suggestion in suggestions:
        payload = suggestion.to_dict()
        assert set(payload) == {
            "category",
            "issue",
            "guidance",
            "severity",
        }
        assert isinstance(payload["guidance"], str)
        assert payload["guidance"]
