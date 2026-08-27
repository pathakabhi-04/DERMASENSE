from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.torch_dataset import CVDatasetTorch
from src.inference.pipeline import DermaSenseInferencePipeline
from src.risk.action_mapping import ProductAction
from src.risk.safety_gate import GateDecision


CHECKPOINT = Path(
    "checkpoints/archive/"
    "pad_ufes_c1_partial_finetune_seed42_best.pt"
)

PREDICTIONS = Path(
    "analysis/product_eval/"
    "c1_f1_test_predictions.csv"
)

HIGH_RISK = {
    "BCC",
    "SCC",
    "MEL",
}


def build_pipeline():
    return DermaSenseInferencePipeline.from_checkpoint(
        CHECKPOINT,
        device="cpu",
    )


def build_test_index():
    """
    Build image_id -> dataset index mapping from the canonical
    PAD-UFES test dataset.
    """
    dataset = CVDatasetTorch(
        dataset_id="pad_ufes",
        split="test",
        verify_images=True,
    )

    index = {}

    for i in range(len(dataset)):
        sample = dataset[i]["sample"]
        index[sample.image_id] = i

    return dataset, index


def test_all_known_dangerous_cases_are_intercepted():
    """
    Every C1 high-risk -> MONITOR case identified by the Phase 4
    safety analysis must reach REVIEW through the actual pipeline.
    """

    predictions = pd.read_csv(PREDICTIONS)

    required_columns = {
        "image_id",
        "true_class",
        "c1_pred",
    }

    missing = required_columns - set(predictions.columns)

    assert not missing, (
        "Prediction artifact is missing columns: "
        f"{sorted(missing)}"
    )

    dangerous = predictions[
        predictions["true_class"].isin(HIGH_RISK)
        & ~predictions["c1_pred"].isin(HIGH_RISK)
        & predictions["c1_pred"].isin({"ACK", "NEV", "SEK"})
    ].copy()

    # The product action mapping defines ACK as EVALUATE_SOON,
    # while NEV/SEK map to MONITOR. Therefore only the latter
    # constitute the dangerous high-risk -> MONITOR set.
    dangerous = dangerous[
        dangerous["c1_pred"].isin({"NEV", "SEK"})
    ]

    assert len(dangerous) == 7

    dataset, index = build_test_index()
    pipeline = build_pipeline()

    for row in dangerous.itertuples(index=False):
        assert row.image_id in index, (
            f"Could not find image in PAD-UFES test dataset: "
            f"{row.image_id}"
        )

        result = pipeline.predict(
            dataset[index[row.image_id]]["image"]
        )

        # The inference pipeline must preserve the native classifier
        # prediction discovered during Phase 4 analysis.
        assert result.predicted_class == row.c1_pred, (
            f"Native prediction changed for {row.image_id}: "
            f"artifact={row.c1_pred}, "
            f"pipeline={result.predicted_class}"
        )

        # The dangerous native prediction must map to MONITOR.
        assert result.product_action is ProductAction.MONITOR, (
            f"Unexpected product action for {row.image_id}: "
            f"{result.product_action}"
        )

        # Phase 4 must intercept it.
        assert result.gate_decision is GateDecision.REVIEW, (
            f"Dangerous case was not routed to review: "
            f"{row.image_id}"
        )

        assert result.requires_review is True


def test_known_dangerous_cases_are_not_reclassified():
    """
    The safety gate must not modify the native diagnosis.

    It only changes the downstream product decision.
    """

    predictions = pd.read_csv(PREDICTIONS)

    dangerous = predictions[
        predictions["true_class"].isin(HIGH_RISK)
        & predictions["c1_pred"].isin({"NEV", "SEK"})
    ].copy()

    assert len(dangerous) == 7

    dataset, index = build_test_index()
    pipeline = build_pipeline()

    for row in dangerous.itertuples(index=False):
        result = pipeline.predict(
            dataset[index[row.image_id]]["image"]
        )

        assert result.predicted_class == row.c1_pred

        if row.c1_pred in {"NEV", "SEK"}:
            assert result.product_action is ProductAction.MONITOR
            assert result.requires_review is True
