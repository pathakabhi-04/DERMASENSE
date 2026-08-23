from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


class ModelError(ValueError):
    """Raised when the model is configured incorrectly."""


PAD_UFES_CLASSES = (
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
)

ISIC2019_CLASSES = (
    "AK",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "SCC",
    "VASC",
)


@dataclass(frozen=True)
class NativeClassifierConfig:
    """
    Configuration for the DermaSense native-diagnosis classifier.
    """

    backbone: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.backbone != "resnet18":
            raise ModelError(
                f"Unsupported backbone: {self.backbone!r}. "
                "Only 'resnet18' is currently supported."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ModelError(
                "dropout must be in the range [0, 1)."
            )


class SharedResNet18Backbone(nn.Module):
    """
    Shared visual feature extractor.

    Produces a 512-dimensional representation from a
    224x224 RGB image.
    """

    feature_dim = 512

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet18(weights=weights)

        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ModelError(
                "Expected input shape [B, C, H, W]. "
                f"Got {tuple(x.shape)}."
            )

        if x.shape[1] != 3:
            raise ModelError(
                "Expected 3-channel RGB input. "
                f"Got {x.shape[1]} channels."
            )

        features = self.features(x)

        return torch.flatten(
            features,
            start_dim=1,
        )


class NativeDiagnosisHead(nn.Module):
    """
    Dataset-specific native diagnosis classifier head.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ModelError(
                "num_classes must be greater than one."
            )

        if dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(
                    feature_dim,
                    num_classes,
                ),
            )
        else:
            self.classifier = nn.Linear(
                feature_dim,
                num_classes,
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class DermaSenseNativeClassifier(nn.Module):
    """
    Shared-backbone, dataset-specific native diagnosis model.

    The same visual backbone can be used for both datasets,
    while each dataset retains its own native diagnostic label
    space.

        image
          |
          v
    shared ResNet-18
          |
       features
        /     \
       /       \
      v         v
    PAD       ISIC
    6-way     8-way
    head      head
    """

    def __init__(
        self,
        config: NativeClassifierConfig | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = NativeClassifierConfig()

        self.config = config

        self.backbone = SharedResNet18Backbone(
            pretrained=config.pretrained
        )

        self.pad_ufes_head = NativeDiagnosisHead(
            feature_dim=self.backbone.feature_dim,
            num_classes=len(PAD_UFES_CLASSES),
            dropout=config.dropout,
        )

        self.isic2019_head = NativeDiagnosisHead(
            feature_dim=self.backbone.feature_dim,
            num_classes=len(ISIC2019_CLASSES),
            dropout=config.dropout,
        )

    def extract_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone(x)

    def forward(
        self,
        x: torch.Tensor,
        dataset_id: str,
    ) -> torch.Tensor:
        features = self.extract_features(x)

        if dataset_id == "pad_ufes":
            return self.pad_ufes_head(features)

        if dataset_id == "isic2019":
            return self.isic2019_head(features)

        raise ModelError(
            f"Unknown dataset_id: {dataset_id!r}. "
            "Expected 'pad_ufes' or 'isic2019'."
        )

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return shared backbone features without applying a head.
        """
        return self.extract_features(x)

    def parameter_counts(self) -> dict[str, int]:
        """
        Return parameter counts by model component.
        """

        return {
            "backbone": sum(
                p.numel()
                for p in self.backbone.parameters()
            ),
            "pad_ufes_head": sum(
                p.numel()
                for p in self.pad_ufes_head.parameters()
            ),
            "isic2019_head": sum(
                p.numel()
                for p in self.isic2019_head.parameters()
            ),
            "total": sum(
                p.numel()
                for p in self.parameters()
            ),
        }
