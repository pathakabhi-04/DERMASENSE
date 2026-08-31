"""
CV-1.5 router training data.

Builds the proxy-labeled binary classification set described in
docs/cv1_5_router_spec.md: PAD-UFES-20 images -> pre_framed,
iToBoS 2024 images -> wide_field. This is dataset-identity supervision,
not per-image-verified framing -- see the spec's ground-truth caveat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ("pre_framed", "wide_field")
LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def build_framing_split(
    pad_ufes_csv: Path,
    itobos_csv: Path,
    *,
    max_per_class: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Combine a PAD-UFES split file (-> pre_framed) and an iToBoS split
    file (-> wide_field) into one (image_path, label) table.
    """
    pad = pd.read_csv(pad_ufes_csv)[["image_path"]].copy()
    pad["label"] = "pre_framed"

    ito = pd.read_csv(itobos_csv).drop_duplicates("image_id")
    ito = ito[["image_path"]].copy()
    ito["label"] = "wide_field"

    if max_per_class is not None:
        pad = pad.sample(
            n=min(max_per_class, len(pad)), random_state=seed
        )
        ito = ito.sample(
            n=min(max_per_class, len(ito)), random_state=seed
        )

    return pd.concat([pad, ito], ignore_index=True)


class FramingImageDataset(Dataset):
    """(image, label_index) pairs for the CV-1.5 router."""

    def __init__(
        self,
        table: pd.DataFrame,
        transform: Callable[[Image.Image], torch.Tensor],
    ):
        self.table = table.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[index]
        image_path = REPO_ROOT / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        label_index = LABEL_TO_INDEX[row["label"]]
        return {
            "image": tensor,
            "label": label_index,
            "image_path": row["image_path"],
        }
