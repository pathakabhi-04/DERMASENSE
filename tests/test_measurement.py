"""
CV-7 measurement tests.

Unit tests use a synthetic mask with known, hand-computed geometry (a
filled circle -- exact area/perimeter are known analytically) so the
area/compactness/color math is checked deterministically, no data
needed. The integration test runs the real CV-3 checkpoint on a known
image from the staged sample (skipped if either is unavailable
locally), pinning today's measured output as a regression check --
mirroring tests/test_temporal.py's structure for calibration.py.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.temporal.calibration import RulerCalibration
from src.temporal.measurement import measure_lesion

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ZIP = REPO_ROOT / "data/raw/UQ_zip/866990d01449152d_NIMARE-A11453_A11453.zip"
CHECKPOINT = REPO_ROOT / "checkpoints/cv3_512/best.pt"
ARCHIVE_PREFIX = "866990d01449152d_NIMARE-A11453_A11453/data/Dermoscopic Images/"

# A well-segmented example from the earlier 4-image dev check.
KNOWN_IMAGE = ARCHIVE_PREFIX + "General/General60_Lesion8_visit3.jpg"


def _circle_mask(height=400, width=400, center=(200, 200), radius=80) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.circle(mask, center, radius, 1.0, thickness=-1)
    return mask


CONFIDENT_CALIBRATION = RulerCalibration(
    px_per_mm=250.0, confident=True, num_ticks_detected=5, reason="ok"
)
NOT_CONFIDENT_CALIBRATION = RulerCalibration(
    px_per_mm=None, confident=False, num_ticks_detected=0, reason="no ticks"
)


# ---- unit tests (synthetic, deterministic) ----------------------------


def test_valid_measurement_on_clean_circle():
    mask = _circle_mask()
    image = np.full((400, 400, 3), (150, 160, 200), dtype=np.uint8)

    result = measure_lesion(image, mask, calibration=None)

    assert result.valid
    assert result.reason == "ok"
    # circle area = pi*r^2 = pi*80^2 ~= 20106px, frame = 400*400 = 160000
    assert result.area_fraction == pytest.approx(20106 / 160000, rel=0.02)
    # a perfect circle has compactness ~= 1.0; pixelated-contour
    # perimeter overestimates a discretized circle's true perimeter,
    # so allow for that discretization bias rather than exact roundness
    assert 1.0 <= result.compactness < 1.3
    assert result.mean_lab is not None
    # no calibration provided -> no real-unit fields
    assert result.diameter_mm is None
    assert result.area_mm2 is None


def test_diameter_and_area_mm_present_when_calibration_confident():
    mask = _circle_mask(radius=80)
    image = np.full((400, 400, 3), (150, 160, 200), dtype=np.uint8)

    result = measure_lesion(image, mask, calibration=CONFIDENT_CALIBRATION)

    assert result.valid
    assert result.diameter_mm is not None
    assert result.area_mm2 is not None
    # diameter ~= 2*radius/px_per_mm = 160/250 = 0.64mm
    assert result.diameter_mm == pytest.approx(0.64, rel=0.05)


def test_real_unit_fields_stay_none_when_calibration_not_confident():
    mask = _circle_mask()
    image = np.full((400, 400, 3), (150, 160, 200), dtype=np.uint8)

    result = measure_lesion(image, mask, calibration=NOT_CONFIDENT_CALIBRATION)

    assert result.valid
    assert result.diameter_mm is None
    assert result.area_mm2 is None
    # scale-invariant fields are unaffected by calibration
    assert result.area_fraction is not None
    assert result.compactness is not None


def test_empty_mask_is_invalid():
    mask = np.zeros((400, 400), dtype=np.float32)
    image = np.full((400, 400, 3), (150, 160, 200), dtype=np.uint8)

    result = measure_lesion(image, mask, calibration=CONFIDENT_CALIBRATION)

    assert not result.valid
    assert "empty" in result.reason
    assert result.area_fraction is None
    assert result.diameter_mm is None


def test_only_largest_component_is_measured():
    mask = _circle_mask(center=(100, 100), radius=60)
    mask += _circle_mask(center=(300, 300), radius=10)
    mask = np.clip(mask, 0.0, 1.0)
    image = np.full((400, 400, 3), (150, 160, 200), dtype=np.uint8)

    result = measure_lesion(image, mask, calibration=None)

    assert result.valid
    # if the small blob leaked in, area_fraction would be noticeably larger
    small_blob_area = np.pi * 10 ** 2
    large_blob_area = np.pi * 60 ** 2
    assert result.area_fraction < (large_blob_area + small_blob_area) / (400 * 400)
    assert result.area_fraction == pytest.approx(large_blob_area / (400 * 400), rel=0.05)


def test_mask_is_resized_to_image_resolution():
    # mask at a different resolution than the image, as CV-3's raw
    # 512x512 output is when measuring a full-resolution photo
    mask = _circle_mask(height=512, width=512, center=(256, 256), radius=100)
    image = np.full((1024, 1024, 3), (150, 160, 200), dtype=np.uint8)

    result = measure_lesion(image, mask, calibration=None)

    assert result.valid
    # scaled 2x in each dimension -> area scales 4x; fraction is scale-invariant
    assert result.area_fraction == pytest.approx(
        (np.pi * 100 ** 2) / (512 * 512), rel=0.05
    )


# ---- integration over a real staged image ------------------------------


@pytest.mark.skipif(
    not SOURCE_ZIP.exists() or not CHECKPOINT.exists(),
    reason="UQ source zip or CV-3 checkpoint not available locally",
)
def test_measurement_on_real_image_regression():
    from src.segmentation.inference import load_segmentation_model, predict_mask

    with zipfile.ZipFile(SOURCE_ZIP) as zf, zf.open(KNOWN_IMAGE) as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    device = torch.device("cpu")
    model = load_segmentation_model(CHECKPOINT, device)

    resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(
        rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    ).unsqueeze(0)
    mask = predict_mask(model, tensor, device)

    result = measure_lesion(image, mask, calibration=None)

    assert result.valid
    # pins the roughly ~9-10% area fraction observed during the dev
    # check for this image (analysis/quality/cv7_temporal_data);
    # a wide tolerance since this only guards against gross regressions
    assert 0.01 < result.area_fraction < 0.5
    assert result.compactness > 0
