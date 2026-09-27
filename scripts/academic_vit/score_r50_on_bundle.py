"""R50b: the production CV-4 ResNet-50 scored on the ViT bundle's test images.

docs/academic_vit_spec.md §5 and precondition 4. The comparator for the ViT
CV-4 arm must see identical inputs, so it is re-scored on the pre-resized
bundle (320 px LANCZOS q100 -> CV-4 eval transform Resize((224,224))). It must
come within +/-0.01 of 0.5756 (its score on the originals), or the experiment
stops before training. Also reports melanoma recall with a Wilson interval.

    PYTHONPATH=. python3 scripts/academic_vit/score_r50_on_bundle.py \
        --bundle data/processed/academic_vit_bundle \
        --checkpoint <cv4 weights (model_state_dict)> \
        --out evaluation/academic_vit/r50_on_bundle.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.academic.data import IMAGENET_MEAN, IMAGENET_STD, ISIC2019_CLASSES
from src.academic.metrics import macro_f1, per_class_f1
from src.academic.model import load_cv4_resnet50

ORIGINALS_MACRO_F1 = 0.5756
TOLERANCE = 0.01


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - spread) / den, (centre + spread) / den)


class BundleTest(Dataset):
    def __init__(self, root: Path) -> None:
        df = pd.read_csv(root / "dataset.csv")
        self.df = df[df.split == "test"].reset_index(drop=True)
        self.root = root
        self.tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with Image.open(self.root / r["image"]) as im:
            return self.tf(im.convert("RGB")), ISIC2019_CLASSES.index(r["label"])


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    net = load_cv4_resnet50(a.checkpoint).eval()
    ys, ps = [], []
    for x, y in DataLoader(BundleTest(a.bundle), batch_size=64, num_workers=4):
        ps.append(net(x).argmax(1))
        ys.append(y)
    y, pr = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    mel = ISIC2019_CLASSES.index("MEL")
    k, n = int(((y == mel) & (pr == mel)).sum()), int((y == mel).sum())
    f1 = macro_f1(y, pr, len(ISIC2019_CLASSES))
    result = {
        "n": int(len(y)), "macro_f1": f1, "accuracy": float((y == pr).mean()),
        "f1_per_class": dict(zip(ISIC2019_CLASSES, map(float, per_class_f1(y, pr, len(ISIC2019_CLASSES))))),
        "mel_recall": k / n, "mel_recall_wilson95": wilson(k, n), "mel_n": n,
        "originals_macro_f1": ORIGINALS_MACRO_F1, "delta_vs_originals": f1 - ORIGINALS_MACRO_F1,
        "gate_pass": abs(f1 - ORIGINALS_MACRO_F1) <= TOLERANCE,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
