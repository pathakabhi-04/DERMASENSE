"""
CV-7 ruler calibration tests.

Unit tests use synthetic images with a controlled tick pattern (no
external data needed) to validate the core logic deterministically.
The integration test reads two known real images directly from the
source zip (skipped if it isn't present locally) -- one confirmed
confident, one confirmed failing, identified during the coverage
measurement in analysis/quality/cv7_temporal_data/calibration_result.md.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.temporal.calibration import (
    EXPECTED_PX_PER_MM_RANGE,
    calibrate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ZIP = REPO_ROOT / "data/raw/UQ_zip/866990d01449152d_NIMARE-A11453_A11453.zip"
ARCHIVE_PREFIX = "866990d01449152d_NIMARE-A11453_A11453/data/Dermoscopic Images/"

# Identified during the 200-image coverage measurement
# (analysis/quality/cv7_temporal_data/calibration_result.md).
CONFIDENT_EXAMPLE = ARCHIVE_PREFIX + "HighRisk/HighRisk78_Lesion7_visit2.jpg"
FAILING_EXAMPLE = ARCHIVE_PREFIX + "General/General137_Lesion10_visit1.jpg"


def _synthetic_ruler_image(
    spacing_px: float = 265.0,
    num_ticks: int = 6,
    height: int = 2400,
    width: int = 800,
    tick_x_start: int = 20,
    tick_length: int = 80,
    jitter: float = 0.0,
) -> np.ndarray:
    """
    A blank skin-toned image with evenly-spaced dark horizontal dashes,
    mimicking the ruler's tick pattern at a controlled, known spacing.

    Ticks are placed inside calibrate()'s scanned region
    (y in [0.40*height, 0.90*height], x in [0, 0.15*width]) so these
    synthetic fixtures actually exercise the detector rather than
    silently falling outside its search window.
    """
    image = np.full((height, width, 3), (180, 190, 220), dtype=np.uint8)  # BGR, skin-ish
    rng = np.random.default_rng(0)
    y = height * 0.45
    for _ in range(num_ticks):
        y_int = int(round(y + rng.uniform(-jitter, jitter)))
        cv2.line(
            image,
            (tick_x_start, y_int),
            (tick_x_start + tick_length, y_int),
            (10, 10, 10),
            3,
        )
        y += spacing_px
    return image


# ---- unit tests (synthetic, deterministic) ---------------------------


def test_confident_on_clean_evenly_spaced_ticks():
    image = _synthetic_ruler_image(spacing_px=265.0)
    result = calibrate(image)

    assert result.confident
    assert result.px_per_mm is not None
    assert EXPECTED_PX_PER_MM_RANGE[0] <= result.px_per_mm <= EXPECTED_PX_PER_MM_RANGE[1]
    # allow the Hough/merge pipeline some tolerance vs. the exact synthetic spacing
    assert abs(result.px_per_mm - 265.0) < 15.0


def test_mm_per_pixel_is_inverse_of_px_per_mm():
    image = _synthetic_ruler_image(spacing_px=265.0)
    result = calibrate(image)

    assert result.confident
    assert result.mm_per_pixel() == pytest.approx(1.0 / result.px_per_mm)


def test_not_confident_with_no_ticks():
    blank = np.full((1200, 800, 3), (180, 190, 220), dtype=np.uint8)
    result = calibrate(blank)

    assert not result.confident
    assert result.px_per_mm is None
    assert result.mm_per_pixel() is None


def test_not_confident_with_irregular_spacing():
    height, width = 1200, 800
    image = np.full((height, width, 3), (180, 190, 220), dtype=np.uint8)
    # deliberately uneven gaps, not a real tick pattern -- all within
    # calibrate()'s scanned region (y in [480,1080], x in [0,120])
    for y in [500, 560, 700, 780, 1050]:
        cv2.line(image, (20, y), (100, y), (10, 10, 10), 3)
    result = calibrate(image)

    assert not result.confident
    assert "irregular" in result.reason or "gaps" in result.reason


def test_not_confident_outside_plausible_range():
    # evenly spaced, but at a scale far outside the corroborated range
    image = _synthetic_ruler_image(spacing_px=600.0, num_ticks=4, height=3000)
    result = calibrate(image)

    assert not result.confident
    assert result.px_per_mm is None


def test_result_is_frozen_and_serializable_fields():
    image = _synthetic_ruler_image()
    result = calibrate(image)
    # dataclass fields should be plain, inspectable values
    assert isinstance(result.confident, bool)
    assert isinstance(result.num_ticks_detected, int)
    assert isinstance(result.reason, str)


# ---- integration over real archive images -----------------------------


@pytest.mark.skipif(
    not SOURCE_ZIP.exists(), reason="UQ Longitudinal source zip not available locally"
)
def test_calibration_confident_on_known_real_example():
    with zipfile.ZipFile(SOURCE_ZIP) as zf, zf.open(CONFIDENT_EXAMPLE) as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    result = calibrate(image)

    assert result.confident
    assert EXPECTED_PX_PER_MM_RANGE[0] <= result.px_per_mm <= EXPECTED_PX_PER_MM_RANGE[1]


@pytest.mark.skipif(
    not SOURCE_ZIP.exists(), reason="UQ Longitudinal source zip not available locally"
)
def test_calibration_declines_on_known_failing_example():
    """
    Confirms the module fails LOUDLY (confident=False, px_per_mm=None)
    rather than returning a guessed value on a real image it can't
    reliably read -- the safety property this module exists to
    guarantee, not just an implementation detail.
    """
    with zipfile.ZipFile(SOURCE_ZIP) as zf, zf.open(FAILING_EXAMPLE) as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    result = calibrate(image)

    assert not result.confident
    assert result.px_per_mm is None
