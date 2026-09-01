"""
DermaSense CV-5 explainability.

explain_candidate is the public entry point: given the crop CV-4 saw
plus the already-loaded segmenter and classifier, produces both overlays
(mask contour, Grad-CAM heatmap) and the cross-check evidence between
them. Deliberately NOT wired into DermaSensePipeline.predict()'s hot
path (docs/cv5_explainability_spec.md) -- Grad-CAM needs a backward
pass, meaningfully more expensive than the forward-only evidence CV-3/
CV-6 already attach to every candidate, and its output (overlay images)
does not fit CandidateResult's scalar-evidence pattern. It is invoked
on demand, per candidate, by a caller that already has a PipelineResult.

CandidateResult only stores CV-3's mask as scalar evidence (area
fraction, degenerate flag, border-touch), not the mask array itself, so
the mask is recomputed here -- cheap, deterministic, no wasted training
or randomness involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.transforms import ImageTransformConfig, build_eval_transform
from src.explainability.evidence import gradcam_mask_iou
from src.explainability.gradcam import compute_gradcam
from src.explainability.overlay import gradcam_heatmap_overlay, mask_contour_overlay
from src.inference.crop_normalize import preprocess_crop_for_cv3
from src.inference.native import PAD_CLASSES, NativePredictor
from src.segmentation.inference import predict_mask


@dataclass(frozen=True)
class ExplanationResult:
    """CV-5 output for one candidate: two overlays plus a cross-check."""

    target_class_index: int
    target_class: str
    gradcam_mask_iou: float
    mask_overlay: np.ndarray
    heatmap_overlay: np.ndarray


def explain_candidate(
    crop_bgr: np.ndarray,
    *,
    classifier: NativePredictor,
    segmenter: torch.nn.Module,
    device: torch.device,
    target_class_index: int | None = None,
) -> ExplanationResult:
    """
    Explain one CV-4 diagnosis: what did CV-3 segment, and what did CV-4
    attend to, on the exact crop CV-4 classified.

    Args:
        crop_bgr: the same crop CandidateResult.box_pixels selects from
            the source image (as cv2.imread returns it, BGR).
        classifier: the same NativePredictor CV-4 used.
        segmenter: the same CV-3 model used (pipeline.segmenter).
        device: inference device.
        target_class_index: explain this class; None explains CV-4's
            predicted class.
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    cv3_tensor = preprocess_crop_for_cv3(crop_rgb)
    mask = predict_mask(segmenter, cv3_tensor, device)

    pil_image = Image.fromarray(crop_rgb)
    cv4_tensor = build_eval_transform(ImageTransformConfig())(pil_image)
    cam, target_idx = compute_gradcam(classifier, cv4_tensor, target_class_index)

    return ExplanationResult(
        target_class_index=target_idx,
        target_class=PAD_CLASSES[target_idx],
        gradcam_mask_iou=gradcam_mask_iou(mask, cam),
        mask_overlay=mask_contour_overlay(crop_bgr, mask),
        heatmap_overlay=gradcam_heatmap_overlay(crop_bgr, cam),
    )


__all__ = [
    "ExplanationResult",
    "explain_candidate",
]
