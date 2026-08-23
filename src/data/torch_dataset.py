from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from src.data.dataset import CVDataset, CVSample
from src.data.transforms import (
    ImageTransformConfig,
    build_eval_transform,
    build_train_transform,
)


class TorchDatasetError(RuntimeError):
    """Raised when the PyTorch dataset adapter encounters an invalid state."""


class CVDatasetTorch(Dataset):
    """
    PyTorch adapter over the frozen DermaSense CVDataset.

    This class does not:
      - create splits;
      - modify dataset membership;
      - modify labels;
      - perform risk mapping;
      - alter lesion identity.

    It only converts:

        frozen CVDataset
            ↓
        PIL image
            ↓
        torch.Tensor
            +
        native target
            +
        original CVSample metadata
    """

    def __init__(
        self,
        dataset_id: str,
        split: str,
        transform: Optional[Callable] = None,
        transform_config: Optional[ImageTransformConfig] = None,
        verify_images: bool = True,
    ) -> None:

        self.dataset_id = dataset_id
        self.split = split

        self.base_dataset = CVDataset(
            dataset_id=dataset_id,
            split=split,
            verify_images=verify_images,
        )

        # Use deterministic preprocessing when no explicit transform
        # is supplied.
        if transform is None:
            if split == "train":
                transform = build_train_transform(
                    transform_config
                )
            else:
                transform = build_eval_transform(
                    transform_config
                )

        self.transform = transform

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def target_space(self):
        return self.base_dataset.target_space

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.target_space.class_names

    @property
    def num_classes(self) -> int:
        return self.target_space.num_classes

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        sample: CVSample = self.base_dataset[index]

        image = self.base_dataset.load_image(index)

        image_tensor = self.transform(image)

        if not isinstance(image_tensor, torch.Tensor):
            raise TorchDatasetError(
                "Transform must return a torch.Tensor."
            )

        if image_tensor.ndim != 3:
            raise TorchDatasetError(
                "Expected transformed image with shape "
                "(C, H, W). "
                f"Got {tuple(image_tensor.shape)}"
            )

        if image_tensor.shape[0] != 3:
            raise TorchDatasetError(
                "Expected 3-channel RGB tensor. "
                f"Got {image_tensor.shape[0]} channels."
            )

        if not torch.isfinite(image_tensor).all():
            raise TorchDatasetError(
                f"Non-finite values found in transformed image "
                f"at index {index}."
            )

        target = sample.target_index

        if not isinstance(target, int):
            raise TorchDatasetError(
                f"Target must be an integer. "
                f"Got {type(target).__name__}."
            )

        if target < 0 or target >= self.num_classes:
            raise TorchDatasetError(
                f"Target index {target} is outside the valid "
                f"range [0, {self.num_classes - 1}]."
            )

        # Verify that target_index still corresponds exactly
        # to the native diagnosis.
        decoded = self.target_space.decode(target)

        if decoded != sample.native_diagnosis:
            raise TorchDatasetError(
                "Target/diagnosis mismatch: "
                f"target={target}, "
                f"decoded={decoded!r}, "
                f"diagnosis={sample.native_diagnosis!r}"
            )

        return {
            "image": image_tensor,
            "target": target,
            "sample": sample,
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_sample(self, index: int) -> CVSample:
        """
        Return the original audited CVSample without
        transforming/loading the image.
        """

        return self.base_dataset[index]

    def get_image_id(self, index: int) -> str:
        return self.base_dataset[index].image_id

    def get_diagnosis(self, index: int) -> str:
        return self.base_dataset[index].native_diagnosis

    def get_target(self, index: int) -> int:
        return self.base_dataset[index].target_index

    def __repr__(self) -> str:
        return (
            f"CVDatasetTorch("
            f"dataset_id={self.dataset_id!r}, "
            f"split={self.split!r}, "
            f"samples={len(self)}, "
            f"num_classes={self.num_classes}"
            f")"
        )