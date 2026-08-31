"""
CV-2 -> CV-3 crop normalization (the interface box between detection and
segmentation).

Given a CV-2 bounding box (normalized YOLO xywh) and a source image,
produce a crop preprocessed EXACTLY as CV-3's ISIC2018SegmentationDataset
does, so CV-3 receives input in the distribution it was trained on.

Preprocessing must match src/segmentation/dataset.py byte-for-byte:
  - BGR -> RGB (cv2 loads BGR)
  - straight resize to (512, 512) via cv2.INTER_LINEAR (aspect ratio is
    SQUASHED, not preserved -- matching CV-3's training)
  - float32 / 255.0
  - CHW transpose

Any deviation from CV-3's training preprocessing would confound the
interface validation, so this function deliberately mirrors it.

The margin parameter is the single documented knob: it expands the CV-2
box before cropping, adding surrounding context. margin=0.0 crops the
tight box; margin=0.5 expands each side by 50% of the box dimension.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


CV3_INPUT_SIZE = (512, 512)  # (height, width), matches CV-3 training


def normalized_box_to_pixels(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Convert normalized YOLO xywh to pixel xyxy."""
    cx = x_center * img_w
    cy = y_center * img_h
    w = width * img_w
    h = height * img_h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def expand_and_clip_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    margin: float,
    img_w: int,
    img_h: int,
    center_offset_frac: tuple[float, float] = (0.0, 0.0),
) -> tuple[int, int, int, int]:
    """
    Expand a pixel box by `margin` (fraction of box size per side),
    optionally shift its center by center_offset_frac (fraction of box
    size), and clip to image bounds. Returns integer pixel xyxy.

    center_offset_frac exists to simulate CV-2 localization error
    (imperfect centering) during interface validation.
    """
    w = x2 - x1
    h = y2 - y1

    # apply centering error
    dx = center_offset_frac[0] * w
    dy = center_offset_frac[1] * h
    x1 += dx
    x2 += dx
    y1 += dy
    y2 += dy

    # expand by margin per side
    x1 -= margin * w
    x2 += margin * w
    y1 -= margin * h
    y2 += margin * h

    # clip
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img_w, int(round(x2)))
    y2 = min(img_h, int(round(y2)))

    # guard against degenerate crops
    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    return x1, y1, x2, y2


def preprocess_crop_for_cv3(
    crop_rgb: np.ndarray,
) -> torch.Tensor:
    """
    Apply CV-3's exact preprocessing to an RGB crop.

    Input: HxWx3 uint8 RGB.
    Output: [1, 3, 512, 512] float32 tensor.
    """
    height, width = CV3_INPUT_SIZE
    resized = cv2.resize(
        crop_rgb,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    arr = resized.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
    return tensor


def crop_and_normalize(
    image_bgr: np.ndarray,
    box_norm: tuple[float, float, float, float],
    margin: float = 0.25,
    center_offset_frac: tuple[float, float] = (0.0, 0.0),
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """
    Full CV-2 -> CV-3 interface transform.

    Args:
        image_bgr: source image as loaded by cv2.imread (BGR, HxWx3 uint8)
        box_norm: CV-2 box as normalized (x_center, y_center, w, h)
        margin: context expansion fraction per side
        center_offset_frac: simulated localization error (validation only)

    Returns:
        (cv3_input_tensor [1,3,512,512], pixel_box_used xyxy)
    """
    img_h, img_w = image_bgr.shape[:2]

    x1, y1, x2, y2 = normalized_box_to_pixels(
        *box_norm, img_w, img_h
    )
    px = expand_and_clip_box(
        x1, y1, x2, y2, margin, img_w, img_h, center_offset_frac
    )
    cx1, cy1, cx2, cy2 = px

    crop_bgr = image_bgr[cy1:cy2, cx1:cx2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    tensor = preprocess_crop_for_cv3(crop_rgb)
    return tensor, px