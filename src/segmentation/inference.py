"""
CV-3 segmentation inference helpers.

Extracted from scripts/validate_cv3_domain_itobos.py so the inference
pipeline and the analysis scripts share one definition of "load the
CV-3 checkpoint" and "turn a prepared tensor into a binary mask".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.segmentation.model import build_model

DEFAULT_MASK_THRESHOLD = 0.5


def load_segmentation_model(
    weights: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a trained CV-3 U-Net checkpoint in eval mode."""
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    model = build_model().to(device)
    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    threshold: float = DEFAULT_MASK_THRESHOLD,
) -> np.ndarray:
    """
    Run CV-3 on a prepared [1,3,H,W] tensor and return a binary mask.

    Returns an HxW float array of {0.0, 1.0}.
    """
    logits = model(tensor.to(device))
    probs = torch.sigmoid(logits)
    return (probs > threshold).float().squeeze(0).squeeze(0).cpu().numpy()


def mask_evidence(mask: np.ndarray) -> dict:
    """
    Summarize a predicted mask as structured evidence.

    Per docs/cv1_cv4_assembly_spec.md, CV-3's mask is recorded alongside
    the diagnosis (for CV-5 explainability and later lesion morphometry)
    but never reshapes CV-4's input. These are the recorded fields.

    Definitions match proxy_metrics_for_mask in
    scripts/validate_cv3_domain_itobos.py so the numbers stay comparable
    to the CV-3 domain-validation results.
    """
    height, width = mask.shape
    foreground = float(mask.sum())
    area_fraction = foreground / (height * width)
    degenerate = foreground == 0 or area_fraction > 0.95

    border = np.concatenate(
        [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]
    )

    return {
        "mask_area_fraction": area_fraction,
        "mask_degenerate": bool(degenerate),
        "mask_touches_border": bool(border.sum() > 0),
    }
