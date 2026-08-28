from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.segmentation.losses import BCEDiceLoss
from src.segmentation.model import build_model
from src.segmentation.training import (
    evaluate,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
)


def test_set_seed_is_reproducible():
    set_seed(42)
    first = torch.randn(5)

    set_seed(42)
    second = torch.randn(5)

    assert torch.equal(first, second)


def test_resolve_device_cpu():
    device = resolve_device("cpu")

    assert device.type == "cpu"


def test_evaluate_returns_expected_metrics():
    model = build_model(
        base_channels=4,
    )

    images = torch.randn(
        2,
        3,
        64,
        64,
    )

    masks = torch.randint(
        0,
        2,
        (2, 1, 64, 64),
    ).float()

    dataset = TensorDataset(
        images,
        masks,
    )

    # Adapt TensorDataset output to the dictionary expected by evaluate.
    class DictDataset:
        def __len__(self):
            return len(dataset)

        def __getitem__(self, index):
            image, mask = dataset[index]

            return {
                "image": image,
                "mask": mask,
            }

    loader = DataLoader(
        DictDataset(),
        batch_size=2,
    )

    metrics = evaluate(
        model,
        loader,
        BCEDiceLoss(),
        torch.device("cpu"),
    )

    assert set(metrics) == {
        "loss",
        "dice",
        "iou",
    }

    assert torch.isfinite(
        torch.tensor(metrics["loss"])
    )

    assert 0.0 <= metrics["dice"] <= 1.0
    assert 0.0 <= metrics["iou"] <= 1.0


def test_checkpoint_round_trip(tmp_path: Path):
    model = build_model(
        base_channels=4,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    original = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_val_dice=0.75,
        history=[
            {
                "epoch": 1.0,
                "train_loss": 1.0,
                "val_loss": 0.8,
                "val_dice": 0.7,
                "val_iou": 0.6,
                "learning_rate": 1e-3,
            }
        ],
        config={
            "test": True,
        },
    )

    # Change model parameters.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    checkpoint = load_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        device="cpu",
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["best_val_dice"] == 0.75
    assert checkpoint["config"]["test"] is True

    for key, value in model.state_dict().items():
        assert torch.equal(
            value,
            original[key],
        )
