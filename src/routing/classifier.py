"""
CV-1.5 router -- Stage 2 (learned classifier).

Only reached because Stage 1 (src/routing/heuristic.py) failed the
pre-committed >=90%-per-class gate (analysis/quality/cv1_5_router/result.md).
See docs/cv1_5_router_spec.md for the full decision trail and the
proxy-label caveat (framed/wide_field supervision is dataset identity,
not per-image-verified).

ResNet18, ImageNet-pretrained, fine-tuned as a binary pre_framed /
wide_field classifier.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.data.transforms import (
    ImageTransformConfig,
    build_eval_transform,
)
from src.routing.dataset import CLASS_NAMES


def build_router_model(*, pretrained: bool = True) -> nn.Module:
    """ResNet18 with its head replaced for 2-class framing classification."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def load_router_checkpoint(
    checkpoint_path: str, device: torch.device
) -> nn.Module:
    model = build_router_model(pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def route_image(
    image_bgr: np.ndarray,
    model: nn.Module,
    device: torch.device,
    *,
    transform_config: ImageTransformConfig | None = None,
) -> str:
    """
    Classify a BGR image (as loaded by cv2.imread) as pre_framed or
    wide_field. Mirrors src.routing.heuristic.route_image's signature
    shape for interchangeability between Stage 1 and Stage 2.
    """
    from PIL import Image

    transform = build_eval_transform(transform_config)
    rgb = image_bgr[:, :, ::-1]
    pil_image = Image.fromarray(rgb)
    tensor = transform(pil_image).unsqueeze(0).to(device)

    logits = model(tensor)
    predicted_index = int(torch.argmax(logits, dim=1).item())
    return CLASS_NAMES[predicted_index]
