"""CV-3 test Dice for the ResNet-50 U-Net (spec Section 4).

The same metric as scripts/evaluate_cv3.py on main: per-image Dice at
threshold 0.5 (smooth 1) at 512x512, averaged over the 260 test images.

    PYTHONPATH=. python3 scripts/academic_r50/evaluate_cv3.py \
        --data-root /workspace/dermasense_academic/data/academic_r50_bundle \
        --run-dir   /workspace/dermasense_academic/runs/academic_r50/S_seed42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.academic.data import SegTestDataset
from src.academic.metrics import per_image_dice, per_image_iou
from src.academic.model import AcademicModel


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    model = AcademicModel("S", pretrained=False, seed=ck["seed"], backbone=cfg.get("backbone", "resnet50"),
                          image_size=cfg.get("image_size", 512))
    model.load_state_dict(ck["model"])
    model.to(dev).eval()

    rows = []
    for b in DataLoader(SegTestDataset(args.data_root, image_size=512, split="test"), batch_size=16, num_workers=4):
        logits = model(b["image"].to(dev))["seg"]
        masks = b["mask"].to(dev)
        for i, (d, j) in enumerate(zip(per_image_dice(logits, masks).tolist(), per_image_iou(logits, masks).tolist())):
            rows.append({"image_id": b["image_id"][i], "dice": d, "iou": j})
    df = pd.DataFrame(rows)
    df.to_csv(args.run_dir / "test_isic2018_per_image.csv", index=False)
    summary = {"n": len(df), "dice_mean": float(df.dice.mean()), "iou_mean": float(df.iou.mean()),
               "dice_median": float(df.dice.median()), "best_epoch": ck["epoch"]}
    (args.run_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
