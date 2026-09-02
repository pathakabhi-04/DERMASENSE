"""
CV-7 assembled pipeline tests.

Two layers, mirroring tests/test_pipeline_assembly.py:

1. Unit tests on TemporalResult's contract shape and input validation
   -- no checkpoint needed.
2. An integration test over a real CV-3 checkpoint and two real staged
   images of the same lesion, asserting invariants (not exact values),
   skipped when the checkpoint or source zip is unavailable locally.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.temporal.delta import TemporalVerdict
from src.temporal.pipeline import TemporalPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ZIP = REPO_ROOT / "data/raw/UQ_zip/866990d01449152d_NIMARE-A11453_A11453.zip"
CHECKPOINT = REPO_ROOT / "checkpoints/cv3_512/best.pt"
ARCHIVE_PREFIX = "866990d01449152d_NIMARE-A11453_A11453/data/Dermoscopic Images/"

# Same lesion, two consecutive visits -- General60 was already confirmed
# well-segmented during measurement.py's validation.
EARLIER_VISIT = ARCHIVE_PREFIX + "General/General60_Lesion8_visit1.jpg"
LATER_VISIT = ARCHIVE_PREFIX + "General/General60_Lesion8_visit3.jpg"

CHECKPOINT_PRESENT = CHECKPOINT.exists()
ZIP_PRESENT = SOURCE_ZIP.exists()


def test_assess_pair_rejects_non_array_input():
    pipeline = TemporalPipeline.__new__(TemporalPipeline)  # bypass __init__, no model needed
    with pytest.raises(TypeError):
        pipeline.assess_pair("not an image", "also not an image")


def test_assess_pair_rejects_wrong_shape():
    pipeline = TemporalPipeline.__new__(TemporalPipeline)
    grayscale = np.zeros((100, 100), dtype=np.uint8)
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        pipeline.assess_pair(grayscale, color)


def test_to_dict_matches_locked_contract_shape():
    from src.temporal.delta import LesionDelta
    from src.temporal.measurement import LesionMeasurement
    from src.temporal.calibration import RulerCalibration
    from src.temporal.pipeline import TemporalResult

    result = TemporalResult(
        verdict=TemporalVerdict.STABLE,
        magnitude=0.1,
        confidence=1.0,
        per_feature_deltas={"size": None, "border": 0.05, "color": 3.2},
        compared_timestamps=("2024-01-01", "2024-06-01"),
        reason="test",
        earlier_measurement=LesionMeasurement(True, "ok", 0.05, 1.1, (60.0, 10.0, 10.0), None, None),
        later_measurement=LesionMeasurement(True, "ok", 0.06, 1.15, (61.0, 10.0, 10.0), None, None),
        earlier_calibration=RulerCalibration(None, False, 0, "no ticks"),
        later_calibration=RulerCalibration(None, False, 0, "no ticks"),
    )

    payload = result.to_dict()

    assert set(payload.keys()) == {
        "verdict", "magnitude", "confidence", "per_feature_deltas", "compared_timestamps",
    }
    assert payload["verdict"] == "STABLE"
    assert payload["per_feature_deltas"]["size"] is None
    assert payload["compared_timestamps"] == ["2024-01-01", "2024-06-01"]


@pytest.mark.skipif(
    not CHECKPOINT_PRESENT or not ZIP_PRESENT,
    reason="CV-3 checkpoint or UQ source zip not available locally",
)
def test_assess_pair_on_real_images_end_to_end():
    pipeline = TemporalPipeline.from_checkpoint(CHECKPOINT, device="cpu")

    with zipfile.ZipFile(SOURCE_ZIP) as zf:
        images = {}
        for label, path in (("earlier", EARLIER_VISIT), ("later", LATER_VISIT)):
            with zf.open(path) as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            images[label] = cv2.imdecode(data, cv2.IMREAD_COLOR)

    result = pipeline.assess_pair(
        images["earlier"], images["later"],
        earlier_timestamp="visit1", later_timestamp="visit3",
    )

    assert result.verdict in TemporalVerdict
    assert 0.0 <= result.confidence <= 1.0
    assert result.compared_timestamps == ("visit1", "visit3")
    # both images are well-segmented (per measurement_result.md), so
    # border/color deltas should be computable regardless of calibration
    assert result.per_feature_deltas["border"] is not None
    assert result.per_feature_deltas["color"] is not None
    payload = result.to_dict()
    assert payload["verdict"] == result.verdict.value
