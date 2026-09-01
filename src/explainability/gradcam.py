"""
CV-5 Grad-CAM.

Bypasses NativePredictor.predict() deliberately -- it is @torch.no_grad()
and returns only the NativePrediction dataclass, so it cannot supply the
gradient path Grad-CAM needs. This reaches predictor.model directly.

Uses model.backbone.forward_conv_features() (src/models/native_classifier.py)
-- an additive method, not positional-index hooking -- to get the
pre-pool conv feature map, then reproduces the model's own pooling
(F.adaptive_avg_pool2d) so the logits computed here are identical to
NativePredictor.predict()'s (verified: forward_conv_features + pool
reconstructs forward() exactly).

Hardcodes dataset_id="pad_ufes" / PAD_CLASSES, matching
NativePredictor.predict()'s own current hardcoding -- not a new
limitation introduced here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.inference.native import NativePredictor


def compute_gradcam(
    classifier: NativePredictor,
    tensor: torch.Tensor,
    target_class_index: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Compute a Grad-CAM heatmap for one CV-4 input tensor.

    Args:
        classifier: a loaded NativePredictor.
        tensor: CV-4's preprocessed input, [3,224,224] or [1,3,224,224].
        target_class_index: explain this class; None explains the
            predicted (argmax) class.

    Returns:
        (cam, target_class_index) -- cam is a [7,7] float32 array in
        [0,1] (matching the backbone's pre-pool spatial resolution for a
        224x224 input), target_class_index is the class actually
        explained.
    """
    model = classifier.model
    was_training = model.training
    model.eval()

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(classifier.device)

    with torch.enable_grad():
        conv_features = model.backbone.forward_conv_features(tensor)
        conv_features.retain_grad()  # not a leaf tensor

        pooled = F.adaptive_avg_pool2d(conv_features, 1).flatten(1)
        logits = model.pad_ufes_head(pooled)

        if target_class_index is None:
            target_class_index = int(torch.argmax(logits, dim=1).item())

        model.zero_grad(set_to_none=True)
        logits[0, target_class_index].backward()

        gradients = conv_features.grad  # [1,C,H,W]
        activations = conv_features.detach()

        # Standard Grad-CAM: channel importance = global-average-pooled
        # gradient, CAM = ReLU(sum_c weight_c * activation_c).
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze(0).squeeze(0)

        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

    if was_training:
        model.train()

    return cam.detach().cpu().numpy().astype(np.float32), target_class_index
