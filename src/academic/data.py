"""Datasets over the flat, relative academic bundle.

The bundle is built locally by scripts/academic_joint/build_bundle.py and is
the only thing the pod reads. dataset.csv columns:

    source    ham | isic2018 | pad
    split     train | val | test
    image_id
    label     diagnosis string (empty for isic2018)
    lesion_id
    image     path relative to the bundle root
    mask      path relative to the bundle root (empty for pad)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

ISIC2019_CLASSES = ("AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC")
PAD_CLASSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_mask(path: Path) -> Image.Image:
    """Binary mask as an L-mode image with values {0, 255}.

    Always convert("L") then threshold: 402 of the Tschandl PNGs are palette
    mode with an inverted palette, and np.array() on them reads the mask
    inverted without any error (spec Section 7).
    """
    with Image.open(path) as im:
        gray = im.convert("L")
    return gray.point(lambda v: 255 if v > 127 else 0)


def build_transform(image_size: int, train: bool) -> v2.Compose:
    """CV-4 augmentation, applied jointly to image and mask.

    Resize first, then flips (0.5 / 0.5), rotation (15 degrees) and colour
    jitter (0.10, 0.10, 0.10, 0.02) -- the CV-4 values in
    src/data/transforms.py on main. Geometric ops move the mask with the
    image (nearest-neighbour for the mask); colour jitter and normalisation
    touch the image only.
    """
    ops: list = [v2.Resize((image_size, image_size), antialias=True)]
    if train:
        ops += [
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.5),
            v2.RandomRotation(15.0),
            v2.ColorJitter(0.10, 0.10, 0.10, 0.02),
        ]
    ops += [
        v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.float32, "others": None}, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return v2.Compose(ops)


def read_bundle(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "dataset.csv", dtype=str, keep_default_na=False)
    return df


class HamJointDataset(Dataset):
    """HAM10000 image + Tschandl mask + 8-class ISIC 2019 target."""

    def __init__(self, root: Path, split: str, *, image_size: int = 512, train: bool | None = None) -> None:
        self.root = Path(root)
        df = read_bundle(self.root)
        self.df = df[(df.source == "ham") & (df.split == split)].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"no HAM rows for split {split!r}")
        unknown = set(self.df.label) - set(ISIC2019_CLASSES)
        if unknown:
            raise ValueError(f"unexpected labels: {sorted(unknown)}")
        self.targets = [ISIC2019_CLASSES.index(c) for c in self.df.label]
        self.transform = build_transform(image_size, train if train is not None else split == "train")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        with Image.open(self.root / row["image"]) as im:
            image = tv_tensors.Image(v2.functional.pil_to_tensor(im.convert("RGB")))
        mask = tv_tensors.Mask(v2.functional.pil_to_tensor(load_mask(self.root / row["mask"])))
        image, mask = self.transform(image, mask)
        return {
            "image": image,
            "mask": (mask > 0.5).float(),
            "target": self.targets[i],
            "image_id": row["image_id"],
        }


class SegTestDataset(Dataset):
    """ISIC 2018 Task 1 test images + masks (external segmentation test)."""

    def __init__(self, root: Path, *, image_size: int = 512) -> None:
        self.root = Path(root)
        df = read_bundle(self.root)
        self.df = df[(df.source == "isic2018") & (df.split == "test")].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("no isic2018 test rows in bundle")
        self.transform = build_transform(image_size, train=False)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        with Image.open(self.root / row["image"]) as im:
            image = tv_tensors.Image(v2.functional.pil_to_tensor(im.convert("RGB")))
        mask = tv_tensors.Mask(v2.functional.pil_to_tensor(load_mask(self.root / row["mask"])))
        image, mask = self.transform(image, mask)
        return {"image": image, "mask": (mask > 0.5).float(), "image_id": row["image_id"]}


class PadDataset(Dataset):
    """PAD-UFES-20 smartphone images, 6 classes, for the C1 transfer."""

    def __init__(self, root: Path, split: str, *, image_size: int = 224) -> None:
        self.root = Path(root)
        df = read_bundle(self.root)
        self.df = df[(df.source == "pad") & (df.split == split)].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"no PAD rows for split {split!r}")
        self.targets = [PAD_CLASSES.index(c) for c in self.df.label]
        self.transform = build_transform(image_size, train=split == "train")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        with Image.open(self.root / row["image"]) as im:
            image = tv_tensors.Image(v2.functional.pil_to_tensor(im.convert("RGB")))
        return {"image": self.transform(image), "target": self.targets[i], "image_id": row["image_id"]}
