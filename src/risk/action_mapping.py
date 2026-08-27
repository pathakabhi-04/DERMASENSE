"""
DermaSense native diagnosis -> product action mapping.

This module intentionally sits downstream of dataset-native target
definitions. Native diagnoses remain unchanged; this module determines
the product-level action associated with a predicted diagnosis.
"""

from __future__ import annotations

from enum import Enum


class ProductAction(str, Enum):
    """Product-level action categories."""

    URGENT_EVALUATION = "URGENT_EVALUATION"
    EVALUATE_SOON = "EVALUATE_SOON"
    MONITOR = "MONITOR"
    UNKNOWN = "UNKNOWN"


HIGH_RISK_DIAGNOSES = frozenset({"BCC", "SCC", "MEL"})
EVALUATE_SOON_DIAGNOSES = frozenset({"ACK"})
MONITOR_DIAGNOSES = frozenset({"NEV", "SEK"})


def normalize_diagnosis(diagnosis: str) -> str:
    """Normalize a native diagnosis for deterministic lookup."""
    if not isinstance(diagnosis, str):
        raise TypeError(
            f"diagnosis must be a string, got {type(diagnosis).__name__}"
        )

    return diagnosis.strip().upper()


def diagnosis_to_action(diagnosis: str) -> ProductAction:
    """
    Map a native diagnosis to its downstream product action.

    Native diagnostic labels are not modified or collapsed. The mapping
    only determines the product-level action associated with the label.
    """
    diagnosis = normalize_diagnosis(diagnosis)

    if diagnosis in HIGH_RISK_DIAGNOSES:
        return ProductAction.URGENT_EVALUATION

    if diagnosis in EVALUATE_SOON_DIAGNOSES:
        return ProductAction.EVALUATE_SOON

    if diagnosis in MONITOR_DIAGNOSES:
        return ProductAction.MONITOR

    return ProductAction.UNKNOWN


def is_high_risk_diagnosis(diagnosis: str) -> bool:
    """Return whether a native diagnosis belongs to the high-risk group."""
    return normalize_diagnosis(diagnosis) in HIGH_RISK_DIAGNOSES


def is_monitor_action(action: ProductAction | str) -> bool:
    """Return whether an action represents the monitor category."""
    if isinstance(action, str):
        try:
            action = ProductAction(action)
        except ValueError:
            return False

    return action is ProductAction.MONITOR
