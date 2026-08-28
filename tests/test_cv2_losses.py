import torch

from src.segmentation.losses import (
    BCEDiceLoss,
    BCELoss,
    DiceLoss,
    build_loss,
    dice_score,
)


def test_dice_score_is_bounded():
    logits = torch.randn(
        2,
        1,
        32,
        32,
    )

    targets = torch.randint(
        0,
        2,
        (2, 1, 32, 32),
    ).float()

    score = dice_score(
        logits,
        targets,
    )

    assert score.ndim == 0
    assert 0.0 <= score.item() <= 1.0


def test_perfect_prediction_has_high_dice():
    targets = torch.zeros(
        1,
        1,
        16,
        16,
    )

    targets[:, :, 4:12, 4:12] = 1.0

    logits = torch.where(
        targets > 0,
        torch.tensor(10.0),
        torch.tensor(-10.0),
    )

    score = dice_score(
        logits,
        targets,
    )

    assert score.item() > 0.99


def test_dice_loss_is_low_for_good_prediction():
    targets = torch.zeros(
        1,
        1,
        16,
        16,
    )

    targets[:, :, 4:12, 4:12] = 1.0

    logits = torch.where(
        targets > 0,
        torch.tensor(10.0),
        torch.tensor(-10.0),
    )

    loss = DiceLoss()(
        logits,
        targets,
    )

    assert loss.item() < 0.01


def test_bce_dice_loss_is_finite():
    logits = torch.randn(
        2,
        1,
        32,
        32,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        2,
        (2, 1, 32, 32),
    ).float()

    loss = BCEDiceLoss()(
        logits,
        targets,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_loss_rejects_shape_mismatch():
    logits = torch.randn(
        1,
        1,
        32,
        32,
    )

    targets = torch.randn(
        1,
        1,
        16,
        16,
    )

    loss = BCEDiceLoss()

    try:
        loss(logits, targets)
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for shape mismatch"
    )

def test_bce_loss_is_finite_and_backpropagates():
    logits = torch.randn(
        2,
        1,
        32,
        32,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        2,
        (2, 1, 32, 32),
    ).float()

    loss = BCELoss()(
        logits,
        targets,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_build_loss_variants():
    assert isinstance(
        build_loss("bce_dice"),
        BCEDiceLoss,
    )

    assert isinstance(
        build_loss("dice"),
        DiceLoss,
    )

    assert isinstance(
        build_loss("bce"),
        BCELoss,
    )


def test_build_loss_normalizes_name():
    assert isinstance(
        build_loss("  DICE  "),
        DiceLoss,
    )


def test_build_loss_rejects_unknown_name():
    try:
        build_loss("invalid")
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for unknown loss"
    )