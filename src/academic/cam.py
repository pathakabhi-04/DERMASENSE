"""Grad-CAM maps and their scoring against ground truth.

docs/academic_resnet50_three_task_spec.md, Section 5 (with 3.1 for details).
Every map is standard Grad-CAM: channel weights are the spatial mean of the
gradient of a scalar target at one layer, map = ReLU(sum_c w_c * A_c). Only the
target and the layer differ per task:

    CV-4  predicted-class logit                      layer4
    CV-3  sum of mask logits inside predicted mask   encoder layer4 (Seg-Grad-CAM)
    CV-2  lesion logit of one box through RoI-Align  backbone layer2
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F


class _Capture:
    """Forward hook that keeps a layer's output, with grad retained."""

    def __init__(self, module: nn.Module) -> None:
        self.value = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inp, out):
        out.retain_grad()
        self.value = out

    def remove(self) -> None:
        self.handle.remove()


def _cam_from(act: torch.Tensor, grad: torch.Tensor) -> np.ndarray:
    """[1,C,h,w] activation and gradient -> [h,w] non-negative map (unnormalised)."""
    weights = grad.mean(dim=(2, 3), keepdim=True)
    return torch.relu((weights * act).sum(1))[0].detach().float().cpu().numpy()


def cam_classifier(net: nn.Module, x: torch.Tensor, target: int | None = None) -> tuple[np.ndarray, int]:
    """CV-4: torchvision ResNet-50, x [1,3,H,W]. Returns (map on layer4 grid, class)."""
    cap = _Capture(net.layer4)
    try:
        with torch.enable_grad():
            logits = net(x)
            if target is None:
                target = int(logits.argmax(1))
            net.zero_grad(set_to_none=True)
            logits[0, target].backward()
        return _cam_from(cap.value, cap.value.grad), target
    finally:
        cap.remove()


def cam_segmenter(model: nn.Module, x: torch.Tensor, region: torch.Tensor | None = None) -> tuple[np.ndarray | None, torch.Tensor]:
    """CV-3 Seg-Grad-CAM (Vinogradova et al. 2020) on encoder layer4.

    Target = sum of mask logits over `region` (default: the predicted mask,
    prob >= 0.5). Returns (map or None if the region is empty, region).
    """
    cap = _Capture(model.encoder.layer4)
    try:
        with torch.enable_grad():
            logits = model(x)["seg"]
            if region is None:
                region = (torch.sigmoid(logits.detach()) >= 0.5)
            if not region.any():
                return None, region
            model.zero_grad(set_to_none=True)
            (logits * region).sum().backward()
        return _cam_from(cap.value, cap.value.grad), region
    finally:
        cap.remove()


def cam_detector_box(detector: nn.Module, image: torch.Tensor, box_xyxy: torch.Tensor, *, return_native: bool = False):
    """CV-2: Grad-CAM for one box's lesion score, on backbone layer2 (stride 8).

    The box (original-image coordinates) is pushed through the model's own
    transform, backbone + FPN, RoI-Align and box head; the target is the
    lesion-class logit for that box. Returns the map resized to the original
    image size (padding removed), and with return_native also the layer2-grid
    map: (full, native).
    """
    cap = _Capture(detector.backbone.body.layer2)
    try:
        with torch.enable_grad():
            images, _ = detector.transform([image])
            h0, w0 = image.shape[-2:]
            ht, wt = images.image_sizes[0]
            scale = torch.tensor([wt / w0, ht / h0, wt / w0, ht / h0], device=box_xyxy.device)
            box_t = (box_xyxy * scale).view(1, 4)
            feats = detector.backbone(images.tensors)
            pooled = detector.roi_heads.box_roi_pool(feats, [box_t], images.image_sizes)
            logits, _ = detector.roi_heads.box_predictor(detector.roi_heads.box_head(pooled))
            detector.zero_grad(set_to_none=True)
            logits[0, 1].backward()
        cam = _cam_from(cap.value, cap.value.grad)
    finally:
        cap.remove()
    # layer2 grid covers the padded batch tensor at stride 8: upsample, crop, resize back.
    padded_h, padded_w = images.tensors.shape[-2:]
    t = torch.from_numpy(cam)[None, None]
    t = F.interpolate(t, size=(padded_h, padded_w), mode="bilinear", align_corners=False)[..., :ht, :wt]
    t = F.interpolate(t, size=(h0, w0), mode="bilinear", align_corners=False)
    return (t[0, 0].numpy(), cam) if return_native else t[0, 0].numpy()


def upsample(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(cam))[None, None].float()
    return F.interpolate(t, size=size, mode="bilinear", align_corners=False)[0, 0].numpy()


def pointing_hit(cam: np.ndarray, gt: np.ndarray) -> bool:
    """Does the map's maximum fall inside the ground truth? An all-zero map misses."""
    if cam.max() <= 0:
        return False
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return bool(gt[y, x])


def energy_fraction(cam: np.ndarray, gt: np.ndarray) -> float:
    total = float(cam.sum())
    return float(cam[gt].sum() / total) if total > 0 else 0.0


def iou_at_half(cam: np.ndarray, gt: np.ndarray) -> float:
    """As gradcam_mask_iou on main: normalise by max, threshold 0.5, IoU with the mask."""
    peak = cam.max()
    binary = (cam / peak >= 0.5) if peak > 0 else np.zeros_like(gt, dtype=bool)
    union = np.logical_or(binary, gt).sum()
    return float(np.logical_and(binary, gt).sum() / union) if union else 0.0


def box_mask(shape: tuple[int, int], box) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = box
    m = np.zeros((h, w), dtype=bool)
    m[max(0, int(np.floor(y1))):min(h, int(np.ceil(y2))), max(0, int(np.floor(x1))):min(w, int(np.ceil(x2)))] = True
    return m


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation of two maps on the same grid; undefined (constant map) counts as 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant input: handled below
        r = spearmanr(a.ravel(), b.ravel()).statistic
    return 0.0 if r is None or not np.isfinite(r) else float(r)


def randomize(module: nn.Module, seed: int = 42) -> None:
    """Re-initialise every parameterised submodule of `module` (Adebayo et al. 2018)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for m in module.modules():
            if hasattr(m, "reset_parameters"):  # BatchNorm's also resets running stats
                m.reset_parameters()


# ---------------------------------------------------------------- figures

def _jet(v: np.ndarray) -> np.ndarray:
    """[h,w] in [0,1] -> [h,w,3] uint8, a plain jet colormap (no matplotlib on the pod)."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    peak = cam.max()
    heat = _jet(cam / peak if peak > 0 else cam)
    return (rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


def draw_outline(rgb: np.ndarray, mask: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    edge = mask & ~np.pad(mask, 1)[2:, 1:-1] | mask & ~np.pad(mask, 1)[:-2, 1:-1] \
        | mask & ~np.pad(mask, 1)[1:-1, 2:] | mask & ~np.pad(mask, 1)[1:-1, :-2]
    out = rgb.copy()
    out[edge] = color
    return out


def draw_box(rgb: np.ndarray, box, color, width: int = 2) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = np.clip([x1, x2], 0, w - 1)
    y1, y2 = np.clip([y1, y2], 0, h - 1)
    out[y1:y1 + width, x1:x2 + 1] = color
    out[max(y2 - width + 1, 0):y2 + 1, x1:x2 + 1] = color
    out[y1:y2 + 1, x1:x1 + width] = color
    out[y1:y2 + 1, max(x2 - width + 1, 0):x2 + 1] = color
    return out
