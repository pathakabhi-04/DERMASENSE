"""
CV-7 assembled temporal pipeline.

    earlier image (BGR) -> calibrate + CV-3 mask -> measure_lesion --+
                                                                      |--> compute_delta -> TemporalResult
    later image   (BGR) -> calibrate + CV-3 mask -> measure_lesion --+

Mirrors src/inference/orchestrator.py's shape deliberately: load
component models once (`from_checkpoint`), one entry point
(`assess_pair`, analogous to `predict`) that validates its inputs and
returns an immutable result. This is a separate class, not folded into
`DermaSensePipeline`, because CV-7 consumes a PAIR of images of the
same lesion rather than one image -- a different contract shape, not
just an additional stage.

`TemporalResult.to_dict()` matches the `temporal` sub-object of the
locked JSON contract in docs/cv7_temporal_rag_integration_spec.md:
`{verdict, magnitude, confidence, per_feature_deltas, compared_timestamps}`.

One deliberate deviation from that document's example (flagged here the
same way its own two discrepancies are flagged, not silently resolved):
the example shows `per_feature_deltas` as three floats, but `size` is
`None` whenever either visit lacks a confident ruler calibration
(4.0% coverage per image -- see calibration_result.md), and the whole
dict is all-`None` on `NO_PRIOR_DATA`. A `0.0` standing in for "not
measurable" would silently misrepresent "no change" as a measured fact,
which is exactly the failure mode this project's fail-loud pattern
exists to prevent (see calibration.py, measurement.py). Whoever wires
this into CV-8's actual JSON emission should decide there, not here,
how `None` serializes -- this module keeps it explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.segmentation.inference import load_segmentation_model, predict_mask
from src.temporal.calibration import RulerCalibration, calibrate
from src.temporal.delta import LesionDelta, TemporalVerdict, compute_delta
from src.temporal.measurement import LesionMeasurement, measure_lesion

# CV-3's expected input resolution, matching measurement.py's validated
# preprocessing (analysis/quality/cv7_temporal_data/measurement_result.md).
CV3_INPUT_SIZE = 512


@dataclass(frozen=True)
class TemporalResult:
    """One pairwise CV-7 assessment."""

    verdict: TemporalVerdict
    magnitude: float
    confidence: float
    per_feature_deltas: dict[str, float | None]
    compared_timestamps: tuple[str | None, str | None]
    reason: str

    # Full evidence, for debugging/audit -- not part of the locked
    # JSON contract itself, same "record more than the contract
    # requires" pattern as CandidateResult's evidence fields.
    earlier_measurement: LesionMeasurement
    later_measurement: LesionMeasurement
    earlier_calibration: RulerCalibration
    later_calibration: RulerCalibration

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "magnitude": self.magnitude,
            "confidence": self.confidence,
            "per_feature_deltas": self.per_feature_deltas,
            "compared_timestamps": list(self.compared_timestamps),
        }


class TemporalPipeline:
    """
    The assembled CV-7 pipeline: ruler calibration + CV-3 segmentation
    + measurement + delta, run over a pair of same-lesion images.

    Loads CV-3's checkpoint once; run `assess_pair` per lesion pair.
    """

    def __init__(self, *, segmenter: torch.nn.Module, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.segmenter = segmenter

    @classmethod
    def from_checkpoint(
        cls,
        segmentation_checkpoint: str | Path,
        device: str | torch.device = "cpu",
    ) -> "TemporalPipeline":
        device = torch.device(device)
        return cls(
            segmenter=load_segmentation_model(segmentation_checkpoint, device),
            device=device,
        )

    def _measure(self, image_bgr: np.ndarray) -> tuple[LesionMeasurement, RulerCalibration]:
        """Run calibration + CV-3 + measurement on one image."""
        calibration = calibrate(image_bgr)

        resized = cv2.resize(
            image_bgr, (CV3_INPUT_SIZE, CV3_INPUT_SIZE), interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(
            rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        ).unsqueeze(0)
        mask = predict_mask(self.segmenter, tensor, self.device)

        measurement = measure_lesion(image_bgr, mask, calibration)
        return measurement, calibration

    @staticmethod
    def _validate(image_bgr: np.ndarray, label: str) -> None:
        if not isinstance(image_bgr, np.ndarray):
            raise TypeError(f"{label} must be a numpy.ndarray")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"{label} must have shape [H,W,3]")

    def assess_pair(
        self,
        earlier_image_bgr: np.ndarray,
        later_image_bgr: np.ndarray,
        *,
        earlier_timestamp: str | None = None,
        later_timestamp: str | None = None,
    ) -> TemporalResult:
        """
        Assess temporal change between two images of the SAME lesion.

        `earlier_image_bgr`/`later_image_bgr` must already be ordered by
        visit time -- this method does not infer which came first.
        Timestamps are opaque passthrough strings for the JSON contract
        (e.g. ISO dates); they play no role in the measurement itself.
        """
        self._validate(earlier_image_bgr, "earlier_image_bgr")
        self._validate(later_image_bgr, "later_image_bgr")

        earlier_measurement, earlier_calibration = self._measure(earlier_image_bgr)
        later_measurement, later_calibration = self._measure(later_image_bgr)

        delta: LesionDelta = compute_delta(earlier_measurement, later_measurement)

        per_feature_deltas = {
            "size": delta.size_delta_mm,
            "border": delta.border_delta,
            "color": delta.color_delta,
        }

        return TemporalResult(
            verdict=delta.verdict,
            magnitude=delta.magnitude,
            confidence=delta.confidence,
            per_feature_deltas=per_feature_deltas,
            compared_timestamps=(earlier_timestamp, later_timestamp),
            reason=delta.reason,
            earlier_measurement=earlier_measurement,
            later_measurement=later_measurement,
            earlier_calibration=earlier_calibration,
            later_calibration=later_calibration,
        )
