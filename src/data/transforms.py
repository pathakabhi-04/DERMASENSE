from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image
from torchvision import transforms


class TransformError(ValueError):
    """Raised when an invalid transform configuration is supplied."""


@dataclass(frozen=True)
class ImageTransformConfig:
    """
    Configuration for the deterministic CV image pipeline.

    The preprocessing stage is intentionally separate from
    dataset identity, labels, splits, and augmentation.
    """

    image_size: int = 224

    mean: tuple[float, float, float] = (
        0.485,
        0.456,
        0.406,
    )

    std: tuple[float, float, float] = (
        0.229,
        0.224,
        0.225,
    )

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise TransformError(
                "image_size must be greater than zero."
            )

        if len(self.mean) != 3:
            raise TransformError(
                "mean must contain exactly 3 values."
            )

        if len(self.std) != 3:
            raise TransformError(
                "std must contain exactly 3 values."
            )

        if any(value <= 0 for value in self.std):
            raise TransformError(
                "All std values must be greater than zero."
            )


def build_deterministic_transform(
    config: ImageTransformConfig | None = None,
) -> Callable[[Image.Image], torch.Tensor]:
    """
    Build the deterministic image preprocessing pipeline.

    Pipeline:

        PIL RGB image
            ↓
        Resize
            ↓
        CenterCrop
            ↓
        ToTensor
            ↓
        Normalize
            ↓
        model-ready tensor

    No random augmentation is performed here.
    """

    if config is None:
        config = ImageTransformConfig()

    return transforms.Compose(
        [
            transforms.Resize(
                (
                    config.image_size,
                    config.image_size,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.mean,
                std=config.std,
            ),
        ]
    )


def preprocess_image(
    image: Image.Image,
    transform: Callable[[Image.Image], torch.Tensor],
) -> torch.Tensor:
    """
    Convert an arbitrary PIL image into a normalized RGB tensor.
    """

    if not isinstance(image, Image.Image):
        raise TransformError(
            "preprocess_image expects a PIL.Image.Image."
        )

    image = image.convert("RGB")

    tensor = transform(image)

    if not isinstance(tensor, torch.Tensor):
        raise TransformError(
            "Image transform did not return a torch.Tensor."
        )

    if tensor.ndim != 3:
        raise TransformError(
            "Expected image tensor with shape "
            "(C, H, W). "
            f"Got shape: {tuple(tensor.shape)}"
        )

    if tensor.shape[0] != 3:
        raise TransformError(
            "Expected 3 RGB channels. "
            f"Got {tensor.shape[0]} channels."
        )

    return tensor


def build_train_transform(
    config: ImageTransformConfig | None = None,
) -> Callable[[Image.Image], torch.Tensor]:
    """
    Stage-1 training transform.

    For now this intentionally uses the same deterministic
    preprocessing pipeline as evaluation.

    Random augmentation will be introduced separately after
    the baseline preprocessing contract is validated.
    """

    return build_deterministic_transform(config)


def build_eval_transform(
    config: ImageTransformConfig | None = None,
) -> Callable[[Image.Image], torch.Tensor]:
    """
    Validation/test preprocessing.

    This must remain deterministic and contain no random
    augmentation.
    """

    return build_deterministic_transform(config)