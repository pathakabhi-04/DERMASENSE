from src.risk.action_mapping import (
    ProductAction,
    diagnosis_to_action,
)
from src.risk.safety_gate import (
    GateDecision,
    evaluate_prediction,
    should_review,
)


def test_high_risk_mapping():
    assert diagnosis_to_action("BCC") is ProductAction.URGENT_EVALUATION
    assert diagnosis_to_action("SCC") is ProductAction.URGENT_EVALUATION
    assert diagnosis_to_action("MEL") is ProductAction.URGENT_EVALUATION


def test_ack_is_intermediate():
    assert diagnosis_to_action("ACK") is ProductAction.EVALUATE_SOON


def test_monitor_mapping():
    assert diagnosis_to_action("NEV") is ProductAction.MONITOR
    assert diagnosis_to_action("SEK") is ProductAction.MONITOR


def test_monitor_requires_review():
    assert should_review("NEV") is True
    assert should_review("SEK") is True


def test_non_monitor_actions_auto_release():
    assert should_review("BCC") is False
    assert should_review("SCC") is False
    assert should_review("MEL") is False
    assert should_review("ACK") is False


def test_unknown_fails_safe():
    result = evaluate_prediction("UNKNOWN_DIAGNOSIS")

    assert result.predicted_action is ProductAction.UNKNOWN
    assert result.decision is GateDecision.REVIEW
    assert result.requires_review is True


def test_diagnosis_normalization():
    assert diagnosis_to_action(" bcc ") is ProductAction.URGENT_EVALUATION
    assert diagnosis_to_action(" nev ") is ProductAction.MONITOR
