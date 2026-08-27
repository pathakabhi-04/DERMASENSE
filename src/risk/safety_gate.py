"""
DermaSense Phase 4 safety gate.

The safety gate is a downstream product guardrail. It does not attempt
to correct the classifier's native diagnosis. Instead, it prevents a
prediction that would result in the lowest-action product category
(MONITOR) from being automatically released.

Locked Phase 4 policy:

    predicted action == MONITOR
        -> REVIEW

All other known actions proceed normally.

Unknown or malformed actions fail conservatively to REVIEW.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.risk.action_mapping import (
    ProductAction,
    diagnosis_to_action,
)


class GateDecision(str, Enum):
    """Decision made by the safety gate."""

    AUTO_RELEASE = "AUTO_RELEASE"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class SafetyGateResult:
    """Complete result returned by the safety gate."""

    predicted_diagnosis: str
    predicted_action: ProductAction
    decision: GateDecision
    reason: str

    @property
    def requires_review(self) -> bool:
        """Return whether human/review handling is required."""
        return self.decision is GateDecision.REVIEW


def evaluate_action(action: ProductAction | str) -> SafetyGateResult:
    """
    Evaluate a product action through the Phase 4 safety gate.

    MONITOR is always routed to REVIEW.

    UNKNOWN is also routed to REVIEW because the gate must fail safely
    when it cannot establish a valid product action.
    """
    if isinstance(action, str):
        try:
            action = ProductAction(action)
        except ValueError:
            action = ProductAction.UNKNOWN

    if action is ProductAction.MONITOR:
        return SafetyGateResult(
            predicted_diagnosis="",
            predicted_action=action,
            decision=GateDecision.REVIEW,
            reason="MONITOR action requires safety review.",
        )

    if action is ProductAction.UNKNOWN:
        return SafetyGateResult(
            predicted_diagnosis="",
            predicted_action=action,
            decision=GateDecision.REVIEW,
            reason="Unknown product action; fail-safe to review.",
        )

    return SafetyGateResult(
        predicted_diagnosis="",
        predicted_action=action,
        decision=GateDecision.AUTO_RELEASE,
        reason="Known non-monitor action passed safety gate.",
    )


def evaluate_prediction(predicted_diagnosis: str) -> SafetyGateResult:
    """
    Map a native diagnosis to a product action and apply the safety gate.
    """
    action = diagnosis_to_action(predicted_diagnosis)

    if action is ProductAction.MONITOR:
        decision = GateDecision.REVIEW
        reason = "Predicted MONITOR action requires safety review."
    elif action is ProductAction.UNKNOWN:
        decision = GateDecision.REVIEW
        reason = "Unknown diagnosis/action; fail-safe to review."
    else:
        decision = GateDecision.AUTO_RELEASE
        reason = "Known non-monitor action passed safety gate."

    return SafetyGateResult(
        predicted_diagnosis=predicted_diagnosis,
        predicted_action=action,
        decision=decision,
        reason=reason,
    )


def should_review(predicted_diagnosis: str) -> bool:
    """Convenience function returning only the gate decision."""
    return evaluate_prediction(predicted_diagnosis).requires_review
