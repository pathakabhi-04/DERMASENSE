from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ISIC2018SegmentationDataset(Dataset):
    """
    Dataset loader for the frozen DermaSense CV-2 ISIC 2018 Task 1 splits.

    The split CSV is authoritative. Dataset membership and image/mask
    paths are never reconstructed from the raw directory structure.
    """

    def __init__(
        self,
        split_csv: str | Path,
        image_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.split_csv = Path(split_csv)
        self.image_size = image_size

        if not self.split_csv.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {self.split_csv}"
            )

        self.df = pd.read_csv(self.split_csv)

        required_columns = {
            "dataset",
            "image_id",
            "image_path",
            "mask_path",
            "image_domain",
            "cv3_eligible",
        }

        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        if len(self.df) == 0:
            raise ValueError(
                "Split CSV contains no samples"
            )

        if not self.df["image_id"].is_unique:
            raise ValueError(
                "image_id values must be unique within a split"
            )

        if not self.df["cv3_eligible"].astype(bool).all():
            raise ValueError(
                "CV-2 dataset contains ineligible samples"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.df.iloc[index]

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise ValueError(
                f"Could not decode image: {image_path}"
            )

        if mask is None:
            raise ValueError(
                f"Could not decode mask: {mask_path}"
            )

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                "Image/mask dimension mismatch for "
                f"{row['image_id']}: "
                f"{image.shape[:2]} vs {mask.shape[:2]}"
            )

        # OpenCV loads BGR; CV-2 models consume RGB.
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        height, width = self.image_size

        image = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        # Explicit binary representation.
        mask = (mask > 127).astype(np.float32)

        image = (
            image.astype(np.float32) / 255.0
        )

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        mask = np.expand_dims(
            mask,
            axis=0,
        )

        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(image)
            ),
            "mask": torch.from_numpy(
                np.ascontiguousarray(mask)
            ),
            "image_id": str(row["image_id"]),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_domain": str(row["image_domain"]),
        }
