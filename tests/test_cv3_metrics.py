import torch

from src.segmentation.metrics import (
    segmentation_dice,
    segmentation_iou,
    segmentation_metrics,
)


def make_target():
    target = torch.zeros(
        1,
        1,
        16,
        16,
    )

    target[:, :, 4:12, 4:12] = 1.0

    return target


def make_perfect_logits(target):
    return torch.where(
        target > 0,
        torch.tensor(10.0),
        torch.tensor(-10.0),
    )


def test_perfect_prediction_has_dice_near_one():
    target = make_target()
    logits = make_perfect_logits(target)

    dice = segmentation_dice(
        logits,
        target,
    )

    assert dice.item() > 0.99


def test_perfect_prediction_has_iou_near_one():
    target = make_target()
    logits = make_perfect_logits(target)

    iou = segmentation_iou(
        logits,
        target,
    )

    assert iou.item() > 0.99


def test_empty_prediction_and_target_are_valid():
    target = torch.zeros(
        1,
        1,
        16,
        16,
    )

    logits = torch.full_like(
        target,
        -10.0,
    )

    dice = segmentation_dice(
        logits,
        target,
    )

    iou = segmentation_iou(
        logits,
        target,
    )

    assert dice.item() == 1.0
    assert iou.item() == 1.0


def test_metrics_are_bounded():
    target = torch.randint(
        0,
        2,
        (4, 1, 32, 32),
    ).float()

    logits = torch.randn_like(target)

    metrics = segmentation_metrics(
        logits,
        target,
    )

    assert 0.0 <= metrics["dice"] <= 1.0
    assert 0.0 <= metrics["iou"] <= 1.0


def test_metrics_return_python_floats():
    target = make_target()
    logits = make_perfect_logits(target)

    metrics = segmentation_metrics(
        logits,
        target,
    )

    assert isinstance(metrics["dice"], float)
    assert isinstance(metrics["iou"], float)


def test_threshold_changes_prediction():
    target = make_target()

    # Probability ≈ 0.73 inside lesion.
    logits = torch.where(
        target > 0,
        torch.tensor(1.0),
        torch.tensor(-10.0),
    )

    dice_low_threshold = segmentation_dice(
        logits,
        target,
        threshold=0.5,
    )

    dice_high_threshold = segmentation_dice(
        logits,
        target,
        threshold=0.9,
    )

    assert dice_low_threshold.item() > (
        dice_high_threshold.item()
    )


def test_invalid_threshold_is_rejected():
    target = make_target()
    logits = make_perfect_logits(target)

    try:
        segmentation_dice(
            logits,
            target,
            threshold=1.5,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for invalid threshold"
    )
