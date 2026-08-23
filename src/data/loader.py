from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.data.torch_dataset import CVDatasetTorch


class DataLoaderError(RuntimeError):
    """Raised when an invalid DataLoader configuration is supplied."""


@dataclass(frozen=True)
class DataLoaderConfig:
    """
    Runtime configuration for a CV DataLoader.

    Dataset membership, labels, and split identity remain controlled
    by CVDatasetTorch. This configuration only controls batching.
    """

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise DataLoaderError(
                "batch_size must be greater than zero."
            )

        if self.num_workers < 0:
            raise DataLoaderError(
                "num_workers cannot be negative."
            )

def collate_cv_samples(batch: list[dict]) -> dict:
    """
    Collate CVDatasetTorch samples into a PyTorch batch.

    Images and targets are tensor-collated normally.
    CVSample metadata is intentionally preserved as a list because
    CVSample is an audited domain object and should not be converted
    into a tensor/dict by the default PyTorch collator.
    """

    if not batch:
        raise DataLoaderError(
            "Cannot collate an empty batch."
        )

    images = torch.stack(
        [item["image"] for item in batch],
        dim=0,
    )

    targets = torch.tensor(
        [item["target"] for item in batch],
        dtype=torch.long,
    )

    samples = [
        item["sample"]
        for item in batch
    ]

    return {
        "image": images,
        "target": targets,
        "sample": samples,
    }

def build_dataloader(
    dataset: CVDatasetTorch,
    config: Optional[DataLoaderConfig] = None,
) -> DataLoader:
    """
    Build a DataLoader over an already validated CVDatasetTorch.

    Training:
        shuffle=True

    Validation/test:
        shuffle=False

    No dataset membership or split construction happens here.
    """

    if config is None:
        config = DataLoaderConfig()

    if not isinstance(dataset, CVDatasetTorch):
        raise DataLoaderError(
            "build_dataloader expects a CVDatasetTorch instance."
        )

    shuffle = dataset.split == "train"

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        collate_fn=collate_cv_samples,
    )

def build_split_dataloader(
    dataset_id: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    verify_images: bool = True,
) -> DataLoader:
    """
    Convenience factory for one frozen dataset split.
    """

    dataset = CVDatasetTorch(
        dataset_id=dataset_id,
        split=split,
        verify_images=verify_images,
    )

    config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return build_dataloader(
        dataset=dataset,
        config=config,
    )