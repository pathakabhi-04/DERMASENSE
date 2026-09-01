"""
DermaSense assembled CV-1 -> CV-4 inference pipeline.

    image (BGR)
      -> CV-1   quality gate        -> unusable: STOP [QUALITY_REJECTED]
      -> CV-1.5 domain router       -> pre_framed | wide_field
           pre_framed -> one candidate: the whole frame
           wide_field -> CV-2       -> no boxes: STOP [NO_CANDIDATES]
      -> per candidate:
           crop + normalize -> CV-3 mask (EVIDENCE ONLY)
                            -> CV-4 diagnosis + action + safety gate
      -> aggregate                                    [ASSESSED]

See docs/cv1_cv4_assembly_spec.md for the committed design decisions.
The two that shape this module most:

1. CV-3's mask never touches CV-4's input. CV-4 receives the same
   unmodified crop CV-3 received; the mask is recorded as evidence. CV-4
   drives risk, so it must depend on as few upstream failure points as
   possible -- a bad mask must not be able to crop away the tissue CV-4
   needed to see.
2. Non-assessment is a first-class outcome. QUALITY_REJECTED and
   NO_CANDIDATES are distinct from an ASSESSED low-risk result, so a
   lesion the pipeline never saw can never be mistaken for one it
   cleared.

This class deliberately does NOT replace DermaSenseInferencePipeline in
src/inference/pipeline.py, which remains the CV-4-only path with a
predict(tensor) contract that existing tests depend on. This one takes a
raw BGR image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.transforms import ImageTransformConfig, build_eval_transform
from src.inference.crop_normalize import (
    crop_and_normalize,
    pixel_box_to_norm,
)
from src.inference.native import NativePredictor
from src.quality.assessment import QualityResult, assess_image
from src.quality.capture_guidance import (
    CaptureSuggestion,
    build_capture_suggestions,
)
from src.risk.action_mapping import ProductAction
from src.risk.safety_gate import GateDecision
from src.routing.classifier import load_router_checkpoint
from src.routing.classifier import route_image as route_image_classifier
from src.segmentation.inference import (
    load_segmentation_model,
    mask_evidence,
    predict_mask,
)

# CV-2 inference settings, matching the locked values in
# scripts/evaluate_cv2.py so pipeline detections are identical to the
# ones CV-2 was evaluated with.
CV2_CONF_THRESHOLD = 0.25
CV2_NMS_IOU_THRESHOLD = 0.70
CV2_IMAGE_SIZE = 1280

# Crop margin validated by the CV-2 -> CV-3 interface experiment
# (analysis/quality/cv2_cv3_interface/): best mean Dice in the grid.
CROP_MARGIN = 0.25

# Severity ordering for aggregating multiple candidates into one
# image-level action. Higher wins.
_ACTION_SEVERITY = {
    ProductAction.URGENT_EVALUATION: 3,
    ProductAction.EVALUATE_SOON: 2,
    ProductAction.MONITOR: 1,
    ProductAction.UNKNOWN: 0,
}


class PipelineOutcome(str, Enum):
    """
    Terminal outcome of a pipeline run.

    QUALITY_REJECTED and NO_CANDIDATES mean the image was never
    assessed. They exist as distinct values so that "we never looked at
    this" cannot be read as "we looked and it was fine".
    """

    QUALITY_REJECTED = "QUALITY_REJECTED"
    NO_CANDIDATES = "NO_CANDIDATES"
    ASSESSED = "ASSESSED"


@dataclass(frozen=True)
class CandidateResult:
    """One lesion candidate carried through CV-3 and CV-4."""

    candidate_index: int
    box_pixels: tuple[int, int, int, int]
    detection_confidence: float | None

    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    product_action: ProductAction
    gate_decision: GateDecision
    requires_review: bool
    gate_reason: str

    # CV-3 evidence. Recorded, never fed to CV-4.
    mask_area_fraction: float
    mask_degenerate: bool
    mask_touches_border: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "box_x1": self.box_pixels[0],
            "box_y1": self.box_pixels[1],
            "box_x2": self.box_pixels[2],
            "box_y2": self.box_pixels[3],
            "detection_confidence": self.detection_confidence,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "product_action": self.product_action.value,
            "gate_decision": self.gate_decision.value,
            "requires_review": self.requires_review,
            "mask_area_fraction": self.mask_area_fraction,
            "mask_degenerate": self.mask_degenerate,
            "mask_touches_border": self.mask_touches_border,
        }


@dataclass(frozen=True)
class PipelineResult:
    """Complete result of one assembled pipeline run."""

    outcome: PipelineOutcome
    quality: QualityResult
    framing: str | None
    candidates: tuple[CandidateResult, ...] = ()
    suggestions: tuple[CaptureSuggestion, ...] = ()

    @property
    def assessed(self) -> bool:
        return self.outcome is PipelineOutcome.ASSESSED

    @property
    def product_action(self) -> ProductAction:
        """
        Image-level action: the most severe action across candidates.

        An image the pipeline never assessed returns UNKNOWN -- it is
        explicitly not a low-risk result. Callers must check `outcome`
        rather than treating UNKNOWN as "nothing found, all clear".
        """
        if not self.candidates:
            return ProductAction.UNKNOWN

        return max(
            (candidate.product_action for candidate in self.candidates),
            key=lambda action: _ACTION_SEVERITY[action],
        )

    @property
    def requires_review(self) -> bool:
        """
        Whether a human must look at this image.

        Any candidate requiring review escalates the whole image, and a
        run that never reached assessment always requires review -- an
        unassessed image is not a cleared image.
        """
        if self.outcome is not PipelineOutcome.ASSESSED:
            return True

        return any(
            candidate.requires_review for candidate in self.candidates
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "framing": self.framing,
            "quality_usable": self.quality.usable,
            "quality_score": self.quality.quality_score,
            "product_action": self.product_action.value,
            "requires_review": self.requires_review,
            "num_candidates": len(self.candidates),
            "suggestions": [s.to_dict() for s in self.suggestions],
            "candidates": [c.to_dict() for c in self.candidates],
        }


class DermaSensePipeline:
    """
    The assembled CV-1 -> CV-4 pipeline.

    Loads its component models once and runs single images. It does not
    train or modify any checkpoint.
    """

    def __init__(
        self,
        *,
        router: torch.nn.Module,
        segmenter: torch.nn.Module,
        classifier: NativePredictor,
        detector: Any | None = None,
        device: str | torch.device = "cpu",
        crop_margin: float = CROP_MARGIN,
    ):
        self.device = torch.device(device)
        self.router = router
        self.segmenter = segmenter
        self.classifier = classifier
        self.detector = detector
        self.crop_margin = crop_margin
        self._cv4_transform = build_eval_transform(ImageTransformConfig())

    @classmethod
    def from_checkpoints(
        cls,
        *,
        router_checkpoint: str | Path,
        segmentation_checkpoint: str | Path,
        classifier_checkpoint: str | Path,
        detector_weights: str | Path | None = None,
        device: str | torch.device = "cpu",
        crop_margin: float = CROP_MARGIN,
    ) -> "DermaSensePipeline":
        """
        Build a pipeline from component checkpoints.

        `detector_weights` may be omitted to run a pre-framed-only
        pipeline; a wide_field image then terminates as NO_CANDIDATES
        rather than raising, since the wide-field branch is unavailable
        by configuration.
        """
        device = torch.device(device)

        detector = None
        if detector_weights is not None:
            from ultralytics import YOLO

            detector = YOLO(str(detector_weights))

        return cls(
            router=load_router_checkpoint(str(router_checkpoint), device),
            segmenter=load_segmentation_model(segmentation_checkpoint, device),
            classifier=NativePredictor.from_checkpoint(
                classifier_checkpoint, device=device
            ),
            detector=detector,
            device=device,
            crop_margin=crop_margin,
        )

    # ---- stages -------------------------------------------------

    def _detect_candidates(
        self, image_bgr: np.ndarray
    ) -> list[tuple[tuple[float, float, float, float], float]]:
        """Run CV-2. Returns [(pixel xyxy, confidence), ...]."""
        if self.detector is None:
            return []

        results = self.detector.predict(
            source=image_bgr,
            conf=CV2_CONF_THRESHOLD,
            iou=CV2_NMS_IOU_THRESHOLD,
            imgsz=CV2_IMAGE_SIZE,
            device=str(self.device),
            verbose=False,
        )

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()

        return [
            ((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c))
            for b, c in zip(boxes, confidences)
        ]

    def _classify_crop(self, crop_rgb: np.ndarray):
        """
        Run CV-4 on an RGB crop.

        CV-4 expects a 224x224 ImageNet-normalized tensor built from a
        PIL image, which is a different preprocessing path from CV-3's
        512x512 squashed /255 tensor -- hence the separate transform
        rather than reusing CV-3's input.
        """
        pil_image = Image.fromarray(crop_rgb)
        tensor = self._cv4_transform(pil_image)
        return self.classifier.predict(tensor)

    def _run_candidate(
        self,
        image_bgr: np.ndarray,
        box_norm: tuple[float, float, float, float],
        candidate_index: int,
        detection_confidence: float | None,
    ) -> CandidateResult:
        cv3_tensor, px_box = crop_and_normalize(
            image_bgr, box_norm, margin=self.crop_margin
        )

        mask = predict_mask(self.segmenter, cv3_tensor, self.device)
        evidence = mask_evidence(mask)

        # The RGB crop CV-3 saw, recovered from the same pixel box, fed
        # to CV-4 unmodified -- the mask deliberately plays no part here.
        x1, y1, x2, y2 = px_box
        crop_rgb = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        prediction = self._classify_crop(crop_rgb)

        return CandidateResult(
            candidate_index=candidate_index,
            box_pixels=px_box,
            detection_confidence=detection_confidence,
            predicted_class=prediction.predicted_class,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            product_action=prediction.product_action,
            gate_decision=prediction.safety_gate.decision,
            requires_review=prediction.requires_review,
            gate_reason=prediction.safety_gate.reason,
            **evidence,
        )

    # ---- entry point --------------------------------------------

    def predict(self, image_bgr: np.ndarray) -> PipelineResult:
        """Run the full pipeline on one BGR image (as cv2.imread returns)."""
        if not isinstance(image_bgr, np.ndarray):
            raise TypeError("image_bgr must be a numpy.ndarray")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must have shape [H,W,3]")

        quality = assess_image(image_bgr)
        if not quality.usable:
            return PipelineResult(
                outcome=PipelineOutcome.QUALITY_REJECTED,
                quality=quality,
                framing=None,
                suggestions=build_capture_suggestions(quality, None),
            )

        framing = route_image_classifier(image_bgr, self.router, self.device)
        suggestions = build_capture_suggestions(quality, framing)

        img_h, img_w = image_bgr.shape[:2]

        if framing == "pre_framed":
            # The whole frame is the single candidate. Margin expansion
            # clips back to image bounds, so this is the full image.
            candidate_boxes = [((0.5, 0.5, 1.0, 1.0), None)]
        else:
            detections = self._detect_candidates(image_bgr)
            if not detections:
                return PipelineResult(
                    outcome=PipelineOutcome.NO_CANDIDATES,
                    quality=quality,
                    framing=framing,
                    suggestions=suggestions,
                )
            candidate_boxes = [
                (pixel_box_to_norm(*box, img_w, img_h), conf)
                for box, conf in detections
            ]

        candidates = tuple(
            self._run_candidate(image_bgr, box_norm, index, confidence)
            for index, (box_norm, confidence) in enumerate(candidate_boxes)
        )

        return PipelineResult(
            outcome=PipelineOutcome.ASSESSED,
            quality=quality,
            framing=framing,
            candidates=candidates,
            suggestions=suggestions,
        )
