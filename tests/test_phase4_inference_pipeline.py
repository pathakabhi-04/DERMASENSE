from __future__ import annotations

from pathlib import Path

from src.data.torch_dataset import CVDatasetTorch
from src.inference.pipeline import (
    DermaSenseInferencePipeline,
)
from src.risk.action_mapping import ProductAction
from src.risk.safety_gate import GateDecision


CHECKPOINT = Path(
    "checkpoints/archive/"
    "pad_ufes_c1_partial_finetune_seed42_best.pt"
)


def build_pipeline():
    return DermaSenseInferencePipeline.from_checkpoint(
        CHECKPOINT,
        device="cpu",
    )


def test_pipeline_returns_complete_result():
    dataset = CVDatasetTorch(
        dataset_id="pad_ufes",
        split="test",
        verify_images=True,
    )

    pipeline = build_pipeline()

    result = pipeline.predict(
        dataset[0]["image"]
    )

    assert result.predicted_class in {
        "ACK",
        "BCC",
        "MEL",
        "NEV",
        "SCC",
        "SEK",
    }

    assert 0.0 <= result.confidence <= 1.0

    assert abs(
        sum(result.probabilities.values()) - 1.0
    ) < 1e-5

    assert isinstance(
        result.product_action,
        ProductAction,
    )

    assert isinstance(
        result.gate_decision,
        GateDecision,
    )

    assert result.requires_review == (
        result.gate_decision is GateDecision.REVIEW
    )


def test_monitor_prediction_is_always_reviewed():
    dataset = CVDatasetTorch(
        dataset_id="pad_ufes",
        split="test",
        verify_images=True,
    )

    pipeline = build_pipeline()

    monitor_seen = False

    for index in range(len(dataset)):
        result = pipeline.predict(
            dataset[index]["image"]
        )

        if result.product_action is ProductAction.MONITOR:
            monitor_seen = True

            assert result.gate_decision is (
                GateDecision.REVIEW
            )

            assert result.requires_review is True

    assert monitor_seen


def test_non_monitor_prediction_is_not_reviewed():
    dataset = CVDatasetTorch(
        dataset_id="pad_ufes",
        split="test",
        verify_images=True,
    )

    pipeline = build_pipeline()

    non_monitor_seen = False

    for index in range(len(dataset)):
        result = pipeline.predict(
            dataset[index]["image"]
        )

        if result.product_action is not ProductAction.MONITOR:
            non_monitor_seen = True

            assert result.gate_decision is (
                GateDecision.AUTO_RELEASE
            )

            assert result.requires_review is False

    assert non_monitor_seen
