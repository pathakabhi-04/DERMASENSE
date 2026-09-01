"""
CV-5 overlay rendering.

mask_contour_overlay reuses the exact contour-drawing pattern from
scripts/validate_cv3_domain_itobos.py::draw_overlay (cv2.findContours +
cv2.drawContours), already proven there. gradcam_heatmap_overlay is new
-- no colormap-overlay precedent existed anywhere in this repo.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_CONTOUR_COLOR = (0, 255, 0)  # BGR green, matching the existing precedent
DEFAULT_HEATMAP_ALPHA = 0.4


def mask_contour_overlay(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = DEFAULT_CONTOUR_COLOR,
) -> np.ndarray:
    """Draw CV-3's segmentation mask as a contour on the crop."""
    mask_u8 = (mask * 255).astype(np.uint8)
    if mask_u8.shape[:2] != crop_bgr.shape[:2]:
        mask_u8 = cv2.resize(
            mask_u8,
            (crop_bgr.shape[1], crop_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    overlay = crop_bgr.copy()
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def gradcam_heatmap_overlay(
    crop_bgr: np.ndarray,
    cam: np.ndarray,
    *,
    alpha: float = DEFAULT_HEATMAP_ALPHA,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap (values in [0,1]) onto the crop."""
    height, width = crop_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(
        (np.clip(cam_resized, 0.0, 1.0) * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    return cv2.addWeighted(crop_bgr, 1.0 - alpha, heatmap, alpha, 0)
