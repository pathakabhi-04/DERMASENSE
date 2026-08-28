from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions with BatchNorm and ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Max-pooling followed by a DoubleConv block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(
                in_channels,
                out_channels,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling followed by skip-connection concatenation and DoubleConv.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels // 2 + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:

        x = self.up(x)

        # Handle odd spatial dimensions safely.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        x = torch.cat(
            [skip, x],
            dim=1,
        )

        return self.conv(x)


class UNet(nn.Module):
    """
    Baseline U-Net for binary lesion segmentation.

    Input:
        [B, 3, H, W]

    Output:
        [B, 1, H, W]

    The output is raw logits. Sigmoid should be applied only when
    probabilities are required; the training loss will operate directly
    on logits.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be positive"
            )

        if out_channels <= 0:
            raise ValueError(
                "out_channels must be positive"
            )

        if base_channels <= 0:
            raise ValueError(
                "base_channels must be positive"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        self.inc = DoubleConv(
            in_channels,
            base_channels,
        )

        self.down1 = DownBlock(
            base_channels,
            base_channels * 2,
        )

        self.down2 = DownBlock(
            base_channels * 2,
            base_channels * 4,
        )

        self.down3 = DownBlock(
            base_channels * 4,
            base_channels * 8,
        )

        self.down4 = DownBlock(
            base_channels * 8,
            base_channels * 16,
        )

        self.up1 = UpBlock(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8,
        )

        self.up2 = UpBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
        )

        self.up3 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
        )

        self.up4 = UpBlock(
            base_channels * 2,
            base_channels,
            base_channels,
        )

        self.out = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                "Input must have shape [B,C,H,W]"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {x.shape[1]}"
            )

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out(x)


def build_model(
    *,
    base_channels: int = 32,
) -> UNet:
    """
    Construct the CV-2 baseline segmentation model.
    """

    return UNet(
        in_channels=3,
        out_channels=1,
        base_channels=base_channels,
    )
