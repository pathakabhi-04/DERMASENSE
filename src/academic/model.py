"""ResNet-50 encoder with optional U-Net segmentation and classification heads.

One module serves all three arms of docs/academic_joint_seg_cls_spec.md:

    arm S: encoder -> U-Net decoder -> 1-channel mask logits
    arm C: encoder -> GAP -> linear -> 8 class logits
    arm J: both heads on the same encoder

The same U-Net is the CV-3 model needed by
docs/academic_resnet50_three_task_spec.md (build once, reuse).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50

ARMS = ("S", "C", "J")
NUM_CLASSES = 8

# (input channels from below, skip channels, output channels) per decoder
# block, deepest first. Skips: layer3 (/16), layer2 (/8), layer1 (/4),
# stem (/2), none (/1).
_DECODER = (
    (2048, 1024, 256),
    (256, 512, 128),
    (128, 256, 64),
    (64, 64, 32),
    (32, 0, 16),
)


class ResNet50Encoder(nn.Module):
    """torchvision ResNet-50 split into the five stages a U-Net taps."""

    feature_dim = 2048

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)  # /2, 64
        self.maxpool = net.maxpool
        self.layer1 = net.layer1  # /4, 256
        self.layer2 = net.layer2  # /8, 512
        self.layer3 = net.layer3  # /16, 1024
        self.layer4 = net.layer4  # /32, 2048

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        s0 = self.stem(x)
        s1 = self.layer1(self.maxpool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return [s0, s1, s2, s3, s4]


class _ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class _DecoderBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int) -> None:
        super().__init__()
        self.conv1 = _ConvBNReLU(cin + cskip, cout)
        self.conv2 = _ConvBNReLU(cout, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UNetDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_DecoderBlock(*c) for c in _DECODER)
        self.out = nn.Conv2d(_DECODER[-1][2], 1, kernel_size=1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        s0, s1, s2, s3, s4 = feats
        x = s4
        for block, skip in zip(self.blocks, (s3, s2, s1, s0, None)):
            x = block(x, skip)
        return self.out(x)


class AcademicModel(nn.Module):
    """Encoder plus the heads an arm needs.

    forward() returns a dict with "seg" ([B,1,H,W] logits) and/or
    "cls" ([B,8] logits), depending on the arm.
    """

    def __init__(self, arm: str, *, pretrained: bool = True, seed: int = 0,
                 backbone: str = "resnet50", image_size: int = 512) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        self.arm = arm
        self.backbone = backbone
        if backbone == "resnet50":
            self.encoder = ResNet50Encoder(pretrained=pretrained)
        elif backbone == "vit_b_16":
            if arm != "S":
                raise ValueError("the ViT encoder is only used for arm S (Experiment 3 CV-3)")
            from src.academic.vit import ViTEncoder

            self.encoder = ViTEncoder(image_size=image_size, pretrained=pretrained)
        else:
            raise ValueError(f"unknown backbone {backbone!r}")

        # Heads are initialised from fixed, separate generators so arm C and
        # arm J start from the same classifier weights for a given seed, and
        # S and J from the same decoder weights.
        self.cls_head = None
        self.decoder = None
        if arm in ("C", "J"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.cls_head = nn.Linear(ResNet50Encoder.feature_dim, NUM_CLASSES)
        if arm in ("S", "J"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 1_000)
                self.decoder = UNetDecoder()

    def param_groups(self, encoder_lr: float, rest_lr: float) -> list[dict]:
        """Pretrained trunk at encoder_lr; everything else (for the ViT this
        includes the feature pyramid) at rest_lr."""
        trunk = self.encoder.vit if self.backbone == "vit_b_16" else self.encoder
        trunk_ids = {id(p) for p in trunk.parameters()}
        return [{"params": [p for p in self.parameters() if id(p) in trunk_ids], "lr": encoder_lr},
                {"params": [p for p in self.parameters() if id(p) not in trunk_ids], "lr": rest_lr}]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(x)
        out: dict[str, torch.Tensor] = {}
        if self.cls_head is not None:
            pooled = torch.flatten(F.adaptive_avg_pool2d(feats[-1], 1), 1)
            out["cls"] = self.cls_head(pooled)
        if self.decoder is not None:
            out["seg"] = self.decoder(feats)
        return out


def load_cv4_resnet50(path, map_location="cpu") -> nn.Module:
    """The production CV-4 checkpoint as a plain torchvision ResNet-50 (fc: 8 ISIC 2019 classes).

    main's DermaSenseNativeClassifier stores the backbone as
    backbone.features = Sequential(conv1, bn1, relu, maxpool, layer1..4, avgpool)
    and the ISIC head as isic2019_head.classifier (a Linear, dropout 0).
    """
    names = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state = {}
    for key, value in ckpt["model_state_dict"].items():
        if key.startswith("backbone.features."):
            index, rest = key[len("backbone.features."):].split(".", 1)
            state[f"{names[index]}.{rest}"] = value
        elif key.startswith("isic2019_head.classifier."):
            state["fc." + key[len("isic2019_head.classifier."):]] = value
    net = resnet50(weights=None, num_classes=NUM_CLASSES)
    net.load_state_dict(state, strict=True)
    return net


def mask_to_box(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    """Tight (x0, y0, x1, y1) box around a binary [H,W] mask, or None if empty.

    This is the derived localisation output of spec Section 2.2.5; it is not
    a learned head.
    """
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
