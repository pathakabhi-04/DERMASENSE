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
    Configuration for the DermaSense CV image pipeline.
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

    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.5
    rotation_degrees: float = 15.0

    random_resized_crop_enabled: bool = False
    random_resized_crop_scale_min: float = 0.7
    random_resized_crop_scale_max: float = 1.0

    color_jitter_brightness: float = 0.10
    color_jitter_contrast: float = 0.10
    color_jitter_saturation: float = 0.10
    color_jitter_hue: float = 0.02

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

        probabilities = {
            "horizontal_flip_probability":
                self.horizontal_flip_probability,
            "vertical_flip_probability":
                self.vertical_flip_probability,
        }

        for name, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise TransformError(
                    f"{name} must be between 0 and 1."
                )

        if self.rotation_degrees < 0:
            raise TransformError(
                "rotation_degrees cannot be negative."
            )

        if not 0.0 < self.random_resized_crop_scale_min <= 1.0:
            raise TransformError(
                "random_resized_crop_scale_min must be in (0, 1]."
            )

        if not 0.0 < self.random_resized_crop_scale_max <= 1.0:
            raise TransformError(
                "random_resized_crop_scale_max must be in (0, 1]."
            )

        if (
            self.random_resized_crop_scale_min
            > self.random_resized_crop_scale_max
        ):
            raise TransformError(
                "random_resized_crop_scale_min cannot exceed "
                "random_resized_crop_scale_max."
            )

        jitter_values = {
            "brightness": self.color_jitter_brightness,
            "contrast": self.color_jitter_contrast,
            "saturation": self.color_jitter_saturation,
            "hue": self.color_jitter_hue,
        }

        for name, value in jitter_values.items():
            if value < 0:
                raise TransformError(
                    f"color_jitter_{name} cannot be negative."
                )


def _base_preprocessing(
    config: ImageTransformConfig,
) -> list:
    """
    Deterministic preprocessing shared by train/eval.
    """

    return [
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


def build_deterministic_transform(
    config: ImageTransformConfig | None = None,
) -> Callable[[Image.Image], torch.Tensor]:
    """
    Build deterministic preprocessing.

    Used by validation and test.

    No random augmentation is applied.
    """

    if config is None:
        config = ImageTransformConfig()

    return transforms.Compose(
        _base_preprocessing(config)
    )


def build_train_transform(
    config: ImageTransformConfig | None = None,
) -> Callable[[Image.Image], torch.Tensor]:
    """
    Build the training transform.

    Augmentation is applied only during training.
    """

    if config is None:
        config = ImageTransformConfig()

    augmentation = [
        transforms.RandomHorizontalFlip(
            p=config.horizontal_flip_probability
        ),
        transforms.RandomVerticalFlip(
            p=config.vertical_flip_probability
        ),
    ]

    if config.random_resized_crop_enabled:
        augmentation.append(
            transforms.RandomResizedCrop(
                size=config.image_size,
                scale=(
                    config.random_resized_crop_scale_min,
                    config.random_resized_crop_scale_max,
                ),
            )
        )

    augmentation.extend(
        [
            transforms.RandomRotation(
                degrees=config.rotation_degrees
            ),
            transforms.ColorJitter(
                brightness=config.color_jitter_brightness,
                contrast=config.color_jitter_contrast,
                saturation=config.color_jitter_saturation,
                hue=config.color_jitter_hue,
            ),
        ]
    )

    if not config.random_resized_crop_enabled:
        augmentation.extend(
            _base_preprocessing(config)
        )
    else:
        augmentation.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.mean,
                    std=config.std,
                ),
            ]
        )

    return transforms.Compose(augmentation)


def build_eval_transform(
    config: ImageTransformConfig | None = None,
) -> Callable[[Image.Image], torch.Tensor]:
    """
    Build validation/test preprocessing.

    This is deterministic and contains no random augmentation.
    """

    return build_deterministic_transform(config)


def preprocess_image(
    image: Image.Image,
    transform: Callable[[Image.Image], torch.Tensor],
) -> torch.Tensor:
    """
    Convert a PIL image into a normalized RGB tensor.
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

    if not torch.isfinite(tensor).all():
        raise TransformError(
            "Transform produced non-finite tensor values."
        )

    return tensor