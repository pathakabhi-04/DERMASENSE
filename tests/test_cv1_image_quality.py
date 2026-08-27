from __future__ import annotations

import cv2
import numpy as np

from src.quality import (
    assess_image,
    guidance_for_issue,
)


def good_image() -> np.ndarray:
    rng = np.random.default_rng(42)

    image = np.full(
        (512, 512, 3),
        128,
        dtype=np.uint8,
    )

    noise = rng.normal(
        0,
        35,
        image.shape,
    )

    image = np.clip(
        image.astype(np.float32) + noise,
        0,
        255,
    ).astype(np.uint8)

    return image


def test_good_image_is_usable():
    result = assess_image(
        good_image()
    )

    assert result.usable is True
    assert result.recommended_action == "PROCEED"
    assert result.quality_score > 0.50


def test_low_resolution_is_rejected():
    image = good_image()[:100, :100]

    result = assess_image(
        image,
        minimum_dimension=256,
    )

    assert result.usable is False
    assert "resolution" in {
        issue.type
        for issue in result.issues
    }
    assert result.recommended_action == "RETAKE"

def test_effective_detail_rejects_detail_loss_even_at_large_dimensions():
    image = good_image()

    reduced = cv2.resize(
        image,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    )

    restored = cv2.resize(
        reduced,
        (512, 512),
        interpolation=cv2.INTER_NEAREST,
    )

    result = assess_image(
        restored,
        minimum_dimension=256,
    )

    assert result.usable is False
    assert "resolution" in {
        issue.type
        for issue in result.issues
    }

def test_dark_image_is_rejected():
    image = good_image()
    image[:] = 15

    result = assess_image(image)

    assert result.usable is False
    assert any(
        issue.type == "low_brightness"
        for issue in result.issues
    )


def test_low_contrast_is_detected():
    image = np.full(
        (512, 512, 3),
        128,
        dtype=np.uint8,
    )

    result = assess_image(image)

    assert any(
        issue.type == "low_contrast"
        for issue in result.issues
    )


def test_blur_is_detected():
    image = good_image()

    blurred = cv2.GaussianBlur(
        image,
        (31, 31),
        0,
    )

    result = assess_image(blurred)

    assert any(
        issue.type == "motion_blur"
        for issue in result.issues
    )


def test_guidance_is_deterministic():
    guidance = guidance_for_issue(
        "motion_blur"
    )

    assert "steady" in guidance.lower()


def test_result_serialization():
    result = assess_image(
        good_image()
    )

    payload = result.to_dict()

    assert isinstance(payload, dict)
    assert "usable" in payload
    assert "quality_score" in payload
    assert "issues" in payload
    assert "signals" in payload
    assert "recommended_action" in payload
