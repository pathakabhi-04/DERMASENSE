from pathlib import Path

import numpy as np
import torch

from src.segmentation.dataset import (
    ISIC2018SegmentationDataset,
)


ROOT = Path(__file__).resolve().parents[1]


def test_train_dataset_loads():
    dataset = ISIC2018SegmentationDataset(
        ROOT / "data/splits/isic2018_task1/train.csv",
        image_size=(256, 256),
    )

    assert len(dataset) == 2074


def test_sample_shapes_and_types():
    dataset = ISIC2018SegmentationDataset(
        ROOT / "data/splits/isic2018_task1/train.csv",
        image_size=(256, 256),
    )

    sample = dataset[0]

    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)

    assert sample["image"].shape == (3, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)

    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32


def test_mask_is_binary():
    dataset = ISIC2018SegmentationDataset(
        ROOT / "data/splits/isic2018_task1/train.csv",
        image_size=(256, 256),
    )

    sample = dataset[0]

    values = torch.unique(sample["mask"])

    assert set(values.tolist()).issubset({0.0, 1.0})


def test_image_is_normalized():
    dataset = ISIC2018SegmentationDataset(
        ROOT / "data/splits/isic2018_task1/train.csv",
        image_size=(256, 256),
    )

    sample = dataset[0]

    assert torch.all(sample["image"] >= 0.0)
    assert torch.all(sample["image"] <= 1.0)


def test_metadata_is_preserved():
    dataset = ISIC2018SegmentationDataset(
        ROOT / "data/splits/isic2018_task1/train.csv",
        image_size=(256, 256),
    )

    sample = dataset[0]

    assert sample["image_id"] == "ISIC_0000000"
    assert sample["image_domain"] == "dermoscopic"
