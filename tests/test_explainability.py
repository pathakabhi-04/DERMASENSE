"""
CV-5 explainability tests.

Two layers: pure unit tests on overlay rendering / evidence (no
checkpoints), and an integration test against real checkpoints,
following the same pattern as tests/test_pipeline_assembly.py and
tests/test_uncertainty.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.explainability import explain_candidate
from src.explainability.evidence import gradcam_mask_iou
from src.explainability.gradcam import compute_gradcam
from src.explainability.overlay import gradcam_heatmap_overlay, mask_contour_overlay
from src.inference.native import PAD_CLASSES, NativePredictor
from src.models.native_classifier import (
    DermaSenseNativeClassifier,
    NativeClassifierConfig,
)
from src.segmentation.inference import load_segmentation_model

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
)
SEGMENTATION_CHECKPOINT = REPO_ROOT / "checkpoints/cv3_512/best.pt"
PAD_UFES_TEST = REPO_ROOT / "data/splits/pad_ufes/test.csv"

CHECKPOINTS_PRESENT = (
    CLASSIFIER_CHECKPOINT.exists() and SEGMENTATION_CHECKPOINT.exists()
)


# ---- forward_conv_features: additive-only regression check ----------


def test_forward_conv_features_reconstructs_forward_exactly():
    """
    The whole spec decision (docs/cv5_explainability_spec.md) rests on
    this being additive-only: forward_conv_features() + the model's own
    pooling must reproduce forward()'s existing output exactly, not an
    approximation.
    """
    model = DermaSenseNativeClassifier(
        NativeClassifierConfig(backbone="resnet18", pretrained=False)
    )
    x = torch.randn(2, 3, 224, 224)

    conv_features = model.backbone.forward_conv_features(x)
    assert conv_features.shape == (2, 512, 7, 7)

    reconstructed = F.adaptive_avg_pool2d(conv_features, 1).flatten(1)
    direct = model.backbone(x)

    assert torch.allclose(reconstructed, direct, atol=1e-6)


# ---- overlay rendering (pure) ----------------------------------------


def test_mask_contour_overlay_preserves_shape_and_draws_something():
    crop = np.full((64, 64, 3), 200, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[20:40, 20:40] = 1.0

    overlay = mask_contour_overlay(crop, mask)

    assert overlay.shape == crop.shape
    assert not np.array_equal(overlay, crop)  # something was drawn


def test_mask_contour_overlay_no_op_on_empty_mask():
    crop = np.full((64, 64, 3), 200, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.float32)

    overlay = mask_contour_overlay(crop, mask)

    assert np.array_equal(overlay, crop)


def test_mask_contour_overlay_resizes_mismatched_mask():
    crop = np.full((64, 64, 3), 200, dtype=np.uint8)
    mask = np.zeros((7, 7), dtype=np.float32)  # Grad-CAM-scale mask, wrong for this test but exercises the resize path
    mask[2:5, 2:5] = 1.0

    overlay = mask_contour_overlay(crop, mask)

    assert overlay.shape == crop.shape


def test_gradcam_heatmap_overlay_shape_and_blend():
    crop = np.full((64, 64, 3), 100, dtype=np.uint8)
    cam = np.zeros((7, 7), dtype=np.float32)
    cam[3, 3] = 1.0

    overlay = gradcam_heatmap_overlay(crop, cam)

    assert overlay.shape == crop.shape
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay, crop)


# ---- evidence (pure) --------------------------------------------------


def test_gradcam_mask_iou_perfect_overlap():
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:8, 2:8] = 1.0
    cam = mask.copy()  # same resolution, identical region

    iou = gradcam_mask_iou(mask, cam, cam_threshold=0.5)

    assert iou == pytest.approx(1.0)


def test_gradcam_mask_iou_no_overlap():
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[0:3, 0:3] = 1.0
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[7:10, 7:10] = 1.0

    iou = gradcam_mask_iou(mask, cam, cam_threshold=0.5)

    assert iou == pytest.approx(0.0)


def test_gradcam_mask_iou_handles_resolution_mismatch():
    mask = np.zeros((512, 512), dtype=np.float32)
    mask[100:400, 100:400] = 1.0
    cam = np.ones((7, 7), dtype=np.float32)  # low-res, all-attended

    iou = gradcam_mask_iou(mask, cam)

    assert 0.0 <= iou <= 1.0


def test_gradcam_mask_iou_empty_both_is_zero():
    mask = np.zeros((10, 10), dtype=np.float32)
    cam = np.zeros((7, 7), dtype=np.float32)

    assert gradcam_mask_iou(mask, cam) == 0.0


# ---- integration over real checkpoints --------------------------------


@pytest.mark.skipif(
    not CHECKPOINTS_PRESENT, reason="component checkpoints not available"
)
def test_compute_gradcam_matches_predictor_target_class():
    """
    Grad-CAM's own forward pass (through forward_conv_features + pool +
    head) must pick the same argmax class NativePredictor.predict()
    does on the identical tensor -- confirms the bypass path is
    computing the same thing predict() computes, not a divergent one.
    """
    device = torch.device("cpu")
    classifier = NativePredictor.from_checkpoint(
        CLASSIFIER_CHECKPOINT, device=device
    )

    tensor = torch.randn(3, 224, 224)
    prediction = classifier.predict(tensor)

    _, target_class_index = compute_gradcam(classifier, tensor)
    target_class = PAD_CLASSES[target_class_index]

    assert target_class == prediction.predicted_class


@pytest.mark.skipif(
    not CHECKPOINTS_PRESENT, reason="component checkpoints not available"
)
def test_explain_candidate_end_to_end_on_real_images():
    device = torch.device("cpu")
    classifier = NativePredictor.from_checkpoint(
        CLASSIFIER_CHECKPOINT, device=device
    )
    segmenter = load_segmentation_model(SEGMENTATION_CHECKPOINT, device)

    rows = pd.read_csv(PAD_UFES_TEST).head(3)
    for _, row in rows.iterrows():
        crop_bgr = cv2.imread(str(REPO_ROOT / row["image_path"]))
        assert crop_bgr is not None

        result = explain_candidate(
            crop_bgr, classifier=classifier, segmenter=segmenter, device=device
        )

        assert result.target_class in PAD_CLASSES
        assert 0.0 <= result.gradcam_mask_iou <= 1.0
        assert result.mask_overlay.shape == crop_bgr.shape
        assert result.heatmap_overlay.shape == crop_bgr.shape
        assert result.mask_overlay.dtype == np.uint8
        assert result.heatmap_overlay.dtype == np.uint8
