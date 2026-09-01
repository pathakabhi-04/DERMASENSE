"""
CV-5 evidence: does CV-4's attention agree with CV-3's segmentation?

CV-3 and CV-4 are independent (per docs/cv1_cv4_assembly_spec.md, the
mask never gates CV-4's input), so there is no guarantee the classifier
attended to the region the segmenter identified as the lesion. This is
a genuine, previously-unmeasurable cross-check between two components
that have never been compared to each other before.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_CAM_THRESHOLD = 0.5


def gradcam_mask_iou(
    mask: np.ndarray,
    cam: np.ndarray,
    *,
    cam_threshold: float = DEFAULT_CAM_THRESHOLD,
) -> float:
    """
    IoU between CV-3's segmented region and CV-4's high-activation
    Grad-CAM region. mask and cam are typically different resolutions
    (mask 512x512, cam ~7x7) -- cam is resized to mask's shape for a
    common comparison surface.
    """
    height, width = mask.shape
    cam_resized = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)

    cam_binary = cam_resized >= cam_threshold
    mask_binary = mask > 0.5

    union = np.logical_or(cam_binary, mask_binary).sum()
    if union == 0:
        return 0.0

    intersection = np.logical_and(cam_binary, mask_binary).sum()
    return float(intersection / union)
