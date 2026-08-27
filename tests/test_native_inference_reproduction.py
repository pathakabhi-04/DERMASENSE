from __future__ import annotations

import numpy as np
import torch

from src.data.torch_dataset import CVDatasetTorch
from src.inference.native import NativePredictor


CHECKPOINT = (
    "checkpoints/archive/"
    "pad_ufes_c1_partial_finetune_seed42_best.pt"
)

PAD_CLASSES = (
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
)


def test_native_predictor_reproduces_c1_predictions():
    dataset = CVDatasetTorch(
        dataset_id="pad_ufes",
        split="test",
        verify_images=True,
    )

    predictor = NativePredictor.from_checkpoint(
        CHECKPOINT,
        device="cpu",
    )

    predictions = []
    targets = []

    for index in range(len(dataset)):
        sample = dataset[index]

        result = predictor.predict(
            sample["image"]
        )

        predictions.append(
            PAD_CLASSES.index(
                result.predicted_class
            )
        )

        targets.append(
            sample["target"]
        )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    assert len(predictions) == 352
    assert len(targets) == 352

    expected_confusion_matrix = np.array(
        [
            [84, 14, 0, 0, 7, 4],
            [22, 102, 2, 0, 12, 4],
            [0, 2, 6, 1, 0, 0],
            [2, 2, 0, 27, 0, 2],
            [3, 15, 0, 1, 8, 1],
            [2, 1, 1, 6, 1, 20],
        ],
        dtype=np.int64,
    )

    actual_confusion_matrix = np.zeros(
        (len(PAD_CLASSES), len(PAD_CLASSES)),
        dtype=np.int64,
    )

    for target, prediction in zip(
        targets,
        predictions,
    ):
        actual_confusion_matrix[
            target,
            prediction,
        ] += 1

    np.testing.assert_array_equal(
        actual_confusion_matrix,
        expected_confusion_matrix,
    )


def test_native_prediction_maps_to_product_action():
    predictor = NativePredictor.from_checkpoint(
        CHECKPOINT,
        device="cpu",
    )

    dataset = CVDatasetTorch(
        dataset_id="pad_ufes",
        split="test",
        verify_images=True,
    )

    result = predictor.predict(
        dataset[0]["image"]
    )

    assert result.predicted_class in PAD_CLASSES
    assert 0.0 <= result.confidence <= 1.0

    assert set(result.probabilities) == set(
        PAD_CLASSES
    )

    assert abs(
        sum(result.probabilities.values()) - 1.0
    ) < 1e-5

    assert result.product_action is not None
    assert result.safety_gate is not None
