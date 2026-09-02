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

## CV-7/CV-8 wiring (2026-09-02)

`predict()` optionally accepts `prior_image_bgr` -- a previous visit's
photo of the SAME lesion, if the caller has one. Finding *which* prior
image belongs to which lesion is explicitly the caller's job (a
lesion-tracking/history store), not this pipeline's: this class has no
persistence and does not attempt cross-image lesion re-identification.

Given a prior image, temporal pairing (CV-7, via
`src.temporal.pipeline.TemporalPipeline`) only runs when the CURRENT
image has exactly one candidate. With more than one candidate, which
detected lesion the prior photo corresponds to is genuinely ambiguous,
and this pipeline never guesses at an identity match the same way
calibration.py never guesses at a scale it can't confirm -- pairing is
skipped and flagged (`PRIOR_IMAGE_PAIRING_AMBIGUOUS`), not applied to
an arbitrary candidate. See `_resolve_temporal_pairing`.

CV-8 (`src.risk.convergence.assess_risk`) runs for EVERY candidate
regardless -- it degrades to `temporal=None` gracefully (already part
of its own design), so `CandidateResult.risk_assessment` is always
populated, not just when a prior image was supplied.

## CV-5 wiring (2026-09-02)

`compute_gradcam` (src/explainability/gradcam.py) needs a real
backward pass through the classifier (`torch.enable_grad()`), unlike
CV-6's ensemble evidence which is just extra forward passes -- a
materially different cost. Opt-in via `compute_gradcam=True`
(`__init__`/`from_checkpoints`), same pattern as
`additional_ensemble_checkpoints`: off by default, so existing
behavior and tests are unaffected unless requested.

`gradcam_mask_iou` (src/explainability/evidence.py) has no calibrated
threshold anywhere in this project (docs/cv5_cv6_evidence_architecture.md
records it as a raw cross-check, not a gated signal) -- so, like
`ensemble_probability_distance`/`ensemble_confidence_spread`, it is
recorded on `CandidateResult` as a number for a future consumer, and
does NOT get a fabricated `quality_flags` entry the way
`LOW_CROP_CONTRAST` could (that one reuses an independently validated
cutoff; this one has none to reuse).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.transforms import ImageTransformConfig, build_eval_transform
from src.explainability.evidence import gradcam_mask_iou
from src.explainability.gradcam import compute_gradcam
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
from src.quality.signals import blur_signal, contrast_signal
from src.risk.action_mapping import ProductAction
from src.risk.convergence import RiskAssessment, assess_risk
from src.risk.safety_gate import GateDecision
from src.routing.classifier import load_router_checkpoint
from src.routing.classifier import route_image as route_image_classifier
from src.segmentation.inference import (
    load_segmentation_model,
    mask_evidence,
    predict_mask,
)
from src.temporal.pipeline import TemporalPipeline, TemporalResult
from src.uncertainty.calibration import DEFAULT_TEMPERATURE, apply_temperature
from src.uncertainty.ensemble import ensemble_evidence, load_ensemble

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

    # Crop-quality evidence (docs/cv4_domain_evidence_spec.md). Computed
    # on the exact RGB crop CV-4 receives, general-purpose (recorded for
    # every candidate, not conditioned on predicted class). A disclosure
    # signal, not a gate: low crop_contrast correlates with unreliable
    # out-of-domain BCC/ACK predictions, but a low-contrast crop can
    # still be a genuine faint lesion, so it is never used to drop or
    # reclassify a candidate.
    crop_blur: float
    crop_contrast: float

    # CV-6 evidence (docs/cv6_uncertainty_spec.md). Evidence only -- see
    # that spec's "evidence, not a decision" principle. None when the
    # pipeline was built without ensemble members (opt-in, since it
    # roughly doubles CV-4 inference cost per candidate).
    calibrated_confidence: float
    ensemble_agree: bool | None = None
    ensemble_probability_distance: float | None = None
    ensemble_confidence_spread: float | None = None

    # CV-5 evidence (docs/cv5_cv6_evidence_architecture.md): IoU between
    # CV-3's segmented region and CV-4's Grad-CAM high-activation region
    # -- does the classifier's attention agree with the segmenter's
    # lesion boundary? No calibrated threshold exists for this anywhere
    # in this project, so it is recorded raw, not gated into a flag.
    # None unless the pipeline was built with compute_gradcam=True
    # (opt-in, since it needs a real backward pass).
    gradcam_mask_iou: float | None = None

    # CV-8 convergence (src/risk/convergence.py). Always populated --
    # assess_risk() degrades to temporal=None gracefully, so this is
    # never absent just because no prior image was available.
    risk_assessment: RiskAssessment | None = None

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
            "crop_blur": self.crop_blur,
            "crop_contrast": self.crop_contrast,
            "calibrated_confidence": self.calibrated_confidence,
            "ensemble_agree": self.ensemble_agree,
            "ensemble_probability_distance": self.ensemble_probability_distance,
            "ensemble_confidence_spread": self.ensemble_confidence_spread,
            "gradcam_mask_iou": self.gradcam_mask_iou,
            "risk_assessment": self.risk_assessment.to_dict() if self.risk_assessment else None,
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


def _resolve_temporal_pairing(
    num_candidates: int, prior_image_bgr: np.ndarray | None
) -> tuple[bool, str | None]:
    """
    Decide whether CV-7 temporal pairing should run for this predict()
    call. Pure function, no model access, so this is unit-testable
    without checkpoints -- see module docstring's "CV-7/CV-8 wiring"
    section for the reasoning.

    Returns (should_pair, skip_reason). skip_reason is None whenever
    should_pair is True OR no prior image was supplied at all (nothing
    to explain); it is set only when a prior image WAS supplied but
    pairing couldn't be applied, so CV-8 can record why.
    """
    if prior_image_bgr is None:
        return False, None
    if num_candidates != 1:
        return False, "PRIOR_IMAGE_PAIRING_AMBIGUOUS"
    return True, None


def _candidate_lesion_id(lesion_id: str | None, index: int, num_candidates: int) -> str:
    """
    Resolve the lesion_id CV-8 records for one candidate.

    A caller-supplied lesion_id applies directly only when it's
    unambiguous (exactly one candidate); with multiple candidates in
    one image, a single caller-supplied id can't identify which
    detected lesion it names, so each gets its own suffixed id instead
    of all silently sharing the caller's one id.
    """
    if lesion_id is not None and num_candidates == 1:
        return lesion_id
    if lesion_id is not None:
        return f"{lesion_id}-{index}"
    return f"candidate-{index}"


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
        ensemble_classifiers: list[NativePredictor] | None = None,
        compute_gradcam: bool = False,
        device: str | torch.device = "cpu",
        crop_margin: float = CROP_MARGIN,
        calibration_temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.device = torch.device(device)
        self.router = router
        self.segmenter = segmenter
        self.classifier = classifier
        self.detector = detector
        # CV-6 evidence (docs/cv6_uncertainty_spec.md). Opt-in: None
        # unless additional_ensemble_checkpoints was passed to
        # from_checkpoints, since running extra classifiers roughly
        # doubles CV-4 inference cost per candidate.
        self.ensemble_classifiers = ensemble_classifiers
        # CV-5 evidence (docs/cv5_cv6_evidence_architecture.md). Opt-in:
        # off by default, since Grad-CAM needs a real backward pass
        # through the classifier (torch.enable_grad()), a materially
        # different cost from a forward-only prediction.
        self.compute_gradcam_evidence = compute_gradcam
        self.crop_margin = crop_margin
        self.calibration_temperature = calibration_temperature
        self._cv4_transform = build_eval_transform(ImageTransformConfig())
        # CV-7 (src/temporal/pipeline.py). Reuses the same CV-3
        # segmenter already loaded above -- no extra checkpoint, no
        # extra memory. Always constructed (cost-free until called);
        # whether it actually runs per predict() call depends on
        # whether the caller supplied a prior_image_bgr.
        self.temporal_pipeline = TemporalPipeline(segmenter=self.segmenter, device=self.device)

    @classmethod
    def from_checkpoints(
        cls,
        *,
        router_checkpoint: str | Path,
        segmentation_checkpoint: str | Path,
        classifier_checkpoint: str | Path,
        detector_weights: str | Path | None = None,
        additional_ensemble_checkpoints: tuple[str | Path, ...] | None = None,
        compute_gradcam: bool = False,
        device: str | torch.device = "cpu",
        crop_margin: float = CROP_MARGIN,
        calibration_temperature: float = DEFAULT_TEMPERATURE,
    ) -> "DermaSensePipeline":
        """
        Build a pipeline from component checkpoints.

        `detector_weights` may be omitted to run a pre-framed-only
        pipeline; a wide_field image then terminates as NO_CANDIDATES
        rather than raising, since the wide-field branch is unavailable
        by configuration.

        `additional_ensemble_checkpoints` are CV-4 checkpoints run
        ALONGSIDE `classifier_checkpoint` (not instead of it) to produce
        CV-6 ensemble-disagreement evidence -- e.g. pass the seed123
        checkpoint when `classifier_checkpoint` is seed42. Omitted by
        default (opt-in, doubles CV-4 inference cost per candidate).

        `compute_gradcam` opts into CV-5 evidence (`gradcam_mask_iou`)
        per candidate. Omitted by default -- it needs a real backward
        pass through the classifier, a materially different cost from
        a forward-only prediction.
        """
        device = torch.device(device)

        detector = None
        if detector_weights is not None:
            from ultralytics import YOLO

            detector = YOLO(str(detector_weights))

        ensemble_classifiers = None
        if additional_ensemble_checkpoints is not None:
            ensemble_classifiers = load_ensemble(
                additional_ensemble_checkpoints, device=device
            )

        return cls(
            router=load_router_checkpoint(str(router_checkpoint), device),
            segmenter=load_segmentation_model(segmentation_checkpoint, device),
            ensemble_classifiers=ensemble_classifiers,
            compute_gradcam=compute_gradcam,
            calibration_temperature=calibration_temperature,
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

    def _cv4_tensor(self, crop_rgb: np.ndarray) -> torch.Tensor:
        """
        Build CV-4's input tensor from an RGB crop.

        CV-4 expects a 224x224 ImageNet-normalized tensor built from a
        PIL image, which is a different preprocessing path from CV-3's
        512x512 squashed /255 tensor -- hence the separate transform
        rather than reusing CV-3's input. Shared by the primary
        classifier and any CV-6 ensemble members so every classifier
        sees the identical input.
        """
        pil_image = Image.fromarray(crop_rgb)
        return self._cv4_transform(pil_image)

    def _classify_crop(self, crop_rgb: np.ndarray):
        """Run the primary CV-4 classifier on an RGB crop."""
        return self.classifier.predict(self._cv4_tensor(crop_rgb))

    def _run_candidate(
        self,
        image_bgr: np.ndarray,
        box_norm: tuple[float, float, float, float],
        candidate_index: int,
        detection_confidence: float | None,
        *,
        lesion_id: str,
        temporal: TemporalResult | None = None,
        temporal_skip_reason: str | None = None,
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

        # Crop-quality evidence (docs/cv4_domain_evidence_spec.md):
        # computed on the exact crop CV-4 saw, using CV-1's existing
        # blur/contrast measurements applied at candidate scale rather
        # than whole-image scale. Disclosure only -- never gates.
        crop_blur = blur_signal(crop_rgb).score
        crop_contrast = contrast_signal(crop_rgb).score

        # CV-6 calibration evidence (docs/cv6_uncertainty_spec.md):
        # post-hoc temperature scaling on probabilities only, no logits
        # needed. Class order must match PAD_CLASSES for apply_temperature
        # to scale the right entries -- prediction.probabilities is a
        # dict, so re-derive an ordered vector from it.
        class_names = sorted(prediction.probabilities.keys())
        prob_vector = np.array(
            [[prediction.probabilities[name] for name in class_names]]
        )
        calibrated = apply_temperature(prob_vector, self.calibration_temperature)
        calibrated_confidence = float(calibrated.max())

        # CV-6 ensemble evidence, opt-in (see __init__/from_checkpoints).
        ensemble_fields: dict[str, Any] = {
            "ensemble_agree": None,
            "ensemble_probability_distance": None,
            "ensemble_confidence_spread": None,
        }
        if self.ensemble_classifiers:
            cv4_tensor = self._cv4_tensor(crop_rgb)
            ensemble_predictions = [prediction] + [
                member.predict(cv4_tensor) for member in self.ensemble_classifiers
            ]
            ensemble_fields = ensemble_evidence(ensemble_predictions)

        # CV-5 evidence, opt-in (see __init__/from_checkpoints): does
        # CV-4's Grad-CAM attention agree with CV-3's segmented region?
        # No calibrated threshold exists for this anywhere in this
        # project, so the raw IoU is recorded, not gated into a flag.
        gradcam_iou = None
        if self.compute_gradcam_evidence:
            cam, _ = compute_gradcam(self.classifier, self._cv4_tensor(crop_rgb))
            gradcam_iou = gradcam_mask_iou(mask, cam)

        candidate = CandidateResult(
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
            crop_blur=crop_blur,
            crop_contrast=crop_contrast,
            calibrated_confidence=calibrated_confidence,
            gradcam_mask_iou=gradcam_iou,
            **ensemble_fields,
            **evidence,
        )

        # CV-8 convergence (src/risk/convergence.py). Always run, even
        # with temporal=None -- assess_risk() degrades gracefully, so
        # risk_assessment is populated for every candidate regardless
        # of whether a prior image was available or usable.
        risk_assessment = assess_risk(
            candidate,
            lesion_id=lesion_id,
            temporal=temporal,
            extra_quality_flags=(temporal_skip_reason,) if temporal_skip_reason else (),
        )
        return dataclasses.replace(candidate, risk_assessment=risk_assessment)

    # ---- entry point --------------------------------------------

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        lesion_id: str | None = None,
        prior_image_bgr: np.ndarray | None = None,
        prior_timestamp: str | None = None,
        current_timestamp: str | None = None,
    ) -> PipelineResult:
        """
        Run the full pipeline on one BGR image (as cv2.imread returns).

        `prior_image_bgr`, if given, is a previous visit's photo of the
        SAME lesion -- finding it is the caller's job (a lesion-history
        store this pipeline doesn't own), not this method's. It is only
        actually compared (CV-7) when this image has exactly one
        candidate; see `_resolve_temporal_pairing`. `lesion_id`
        identifies that lesion for CV-8's output and for pairing; with
        no lesion_id and no ambiguity, one is synthesized per candidate.
        """
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

        should_pair, skip_reason = _resolve_temporal_pairing(
            len(candidate_boxes), prior_image_bgr
        )
        temporal_result = None
        if should_pair:
            temporal_result = self.temporal_pipeline.assess_pair(
                prior_image_bgr,
                image_bgr,
                earlier_timestamp=prior_timestamp,
                later_timestamp=current_timestamp,
            )

        candidates = tuple(
            self._run_candidate(
                image_bgr,
                box_norm,
                index,
                confidence,
                lesion_id=_candidate_lesion_id(lesion_id, index, len(candidate_boxes)),
                temporal=temporal_result if should_pair else None,
                temporal_skip_reason=skip_reason,
            )
            for index, (box_norm, confidence) in enumerate(candidate_boxes)
        )

        return PipelineResult(
            outcome=PipelineOutcome.ASSESSED,
            quality=quality,
            framing=framing,
            candidates=candidates,
            suggestions=suggestions,
        )
