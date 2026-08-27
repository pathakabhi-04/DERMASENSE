"""
DermaSense native classifier inference.

This module loads a trained native classifier checkpoint and performs
single-image inference. Product-level safety handling remains downstream
in src.risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.models.native_classifier import (
    DermaSenseNativeClassifier,
    NativeClassifierConfig,
)
from src.risk.action_mapping import (
    ProductAction,
    diagnosis_to_action,
)
from src.risk.safety_gate import (
    SafetyGateResult,
    evaluate_prediction,
)


PAD_CLASSES = (
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
)


@dataclass(frozen=True)
class NativePrediction:
    """Complete native prediction plus downstream safety decision."""

    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    product_action: ProductAction
    safety_gate: SafetyGateResult

    @property
    def requires_review(self) -> bool:
        """Return whether the prediction must be sent for review."""
        return self.safety_gate.requires_review


class NativePredictor:
    """
    Evaluation/inference wrapper around the DermaSense native classifier.

    The predictor:
      1. loads a trained checkpoint,
      2. runs the native classifier,
      3. converts logits to probabilities,
      4. determines the native diagnosis,
      5. maps that diagnosis to a product action,
      6. applies the Phase 4 safety gate.

    It does not modify the checkpoint or train the model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "NativePredictor":
        """Load a C1 native classifier from a checkpoint."""

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_path}"
            )

        model_config = NativeClassifierConfig(
            backbone="resnet50",
            pretrained=False,
            dropout=0.0,
        )

        model = DermaSenseNativeClassifier(
            model_config
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                "Checkpoint must contain a dictionary."
            )

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict"),
        )

        if state_dict is None:
            raise RuntimeError(
                "Could not find model state dict in checkpoint."
            )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        return cls(
            model,
            device=device,
        )

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
    ) -> NativePrediction:
        """
        Run inference on one image.

        Accepted input shapes:
            [C, H, W]
            [1, C, H, W]

        The image must already use the same preprocessing expected
        by the native classifier.
        """

        if not isinstance(image, torch.Tensor):
            raise TypeError(
                "image must be a torch.Tensor"
            )

        if image.ndim == 3:
            image = image.unsqueeze(0)

        if image.ndim != 4:
            raise ValueError(
                "image must have shape [C,H,W] or [1,C,H,W]"
            )

        if image.shape[0] != 1:
            raise ValueError(
                "predict() accepts exactly one image at a time."
            )

        image = image.to(
            self.device,
            non_blocking=True,
        )

        logits = self.model(
            image,
            dataset_id="pad_ufes",
        )

        if logits.ndim != 2 or logits.shape[1] != len(PAD_CLASSES):
            raise RuntimeError(
                "Unexpected classifier output shape: "
                f"{tuple(logits.shape)}"
            )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_index = int(
            torch.argmax(
                probabilities,
                dim=1,
            ).item()
        )

        predicted_class = PAD_CLASSES[
            predicted_index
        ]

        confidence = float(
            probabilities[
                0,
                predicted_index,
            ].item()
        )

        probability_map = {
            class_name: float(
                probabilities[
                    0,
                    class_index,
                ].item()
            )
            for class_index, class_name
            in enumerate(PAD_CLASSES)
        }

        product_action = diagnosis_to_action(
            predicted_class
        )

        safety_gate = evaluate_prediction(
            predicted_class
        )

        return NativePrediction(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probability_map,
            product_action=product_action,
            safety_gate=safety_gate,
        )