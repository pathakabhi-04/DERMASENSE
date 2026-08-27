"""
DermaSense end-to-end inference pipeline.

Pipeline:

    image
      ↓
    native classifier
      ↓
    native diagnosis
      ↓
    product action mapping
      ↓
    Phase 4 safety gate
      ↓
    AUTO_RELEASE / REVIEW

The safety policy is intentionally downstream of the classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.inference.native import (
    NativePrediction,
    NativePredictor,
)
from src.risk.action_mapping import ProductAction
from src.risk.safety_gate import (
    GateDecision,
    SafetyGateResult,
)


@dataclass(frozen=True)
class InferenceResult:
    """Final result exposed by the DermaSense inference pipeline."""

    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    product_action: ProductAction
    gate_decision: GateDecision
    requires_review: bool
    gate_reason: str

    @classmethod
    def from_native_prediction(
        cls,
        prediction: NativePrediction,
    ) -> "InferenceResult":
        """Construct the public inference result."""

        gate: SafetyGateResult = prediction.safety_gate

        return cls(
            predicted_class=prediction.predicted_class,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            product_action=prediction.product_action,
            gate_decision=gate.decision,
            requires_review=gate.requires_review,
            gate_reason=gate.reason,
        )


class DermaSenseInferencePipeline:
    """
    End-to-end native classification + Phase 4 safety pipeline.

    This class does not train or modify the model.
    """

    def __init__(
        self,
        predictor: NativePredictor,
    ):
        self.predictor = predictor

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "DermaSenseInferencePipeline":
        """Create a pipeline from a trained native checkpoint."""

        predictor = NativePredictor.from_checkpoint(
            checkpoint_path,
            device=device,
        )

        return cls(
            predictor
        )

    def predict(
        self,
        image: torch.Tensor,
    ) -> InferenceResult:
        """
        Run the complete DermaSense inference pipeline.

        The image must already use the preprocessing expected by
        the native classifier.
        """

        prediction = self.predictor.predict(
            image
        )

        return InferenceResult.from_native_prediction(
            prediction
        )
