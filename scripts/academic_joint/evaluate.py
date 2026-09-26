"""Test-set evaluation of one finished (arm, seed) run.

Spec Section 6: HAM test macro-F1 (C, J), HAM test Dice (S, J) and ISIC 2018
Task 1 test Dice (S, J). Loads best.pt (selected on val only) and writes
<run>/test_metrics.json, <run>/test_ham_per_image.csv and, for seg arms,
<run>/test_isic2018_per_image.csv.

    PYTHONPATH=. python3 scripts/academic_joint/evaluate.py \
        --run-dir /workspace/dermasense_academic/runs/academic_joint/J_seed42 \
        --data-root /workspace/dermasense_academic/data/academic_joint_bundle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.academic.data import ISIC2019_CLASSES, HamJointDataset, SegTestDataset
from src.academic.metrics import macro_f1, per_class_f1, per_image_dice, per_image_iou
from src.academic.model import AcademicModel, mask_to_box


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=None, help="Smoke tests only.")
    return p.parse_args()


@torch.no_grad()
def run(model, ds, device, batch_size, num_workers, max_batches):
    if max_batches is not None:
        ds = Subset(ds, range(min(len(ds), max_batches * batch_size)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    rows = []
    for batch in loader:
        images = batch["image"].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(images)
        bs = images.shape[0]
        recs = [{"image_id": batch["image_id"][i]} for i in range(bs)]
        if "cls" in out and "target" in batch:
            probs = torch.softmax(out["cls"].float(), 1).cpu()
            for i in range(bs):
                recs[i]["true"] = ISIC2019_CLASSES[int(batch["target"][i])]
                recs[i]["pred"] = ISIC2019_CLASSES[int(probs[i].argmax())]
                for c, name in enumerate(ISIC2019_CLASSES):
                    recs[i][f"p_{name}"] = round(float(probs[i, c]), 5)
        if "seg" in out:
            masks = batch["mask"].to(device)
            dice = per_image_dice(out["seg"], masks).cpu()
            iou = per_image_iou(out["seg"], masks).cpu()
            pred_masks = (torch.sigmoid(out["seg"].float()) >= 0.5).squeeze(1).cpu()
            for i in range(bs):
                recs[i]["dice"] = float(dice[i])
                recs[i]["iou"] = float(iou[i])
                recs[i]["pred_box_xyxy"] = mask_to_box(pred_masks[i])
        rows += recs
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    arm, cfg = ckpt["arm"], ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AcademicModel(arm, pretrained=False, seed=ckpt["seed"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    metrics = {"arm": arm, "seed": ckpt["seed"], "best_epoch": ckpt["epoch"]}

    ham = run(model, HamJointDataset(args.data_root, "test", image_size=cfg["image_size"]),
              device, args.batch_size, args.num_workers, args.max_batches)
    ham.to_csv(args.run_dir / "test_ham_per_image.csv", index=False)
    metrics["ham_test_n"] = len(ham)
    if arm in ("C", "J"):
        y = ham["true"].map(ISIC2019_CLASSES.index).to_numpy()
        yhat = ham["pred"].map(ISIC2019_CLASSES.index).to_numpy()
        metrics["ham_test_macro_f1"] = macro_f1(y, yhat, len(ISIC2019_CLASSES))
        metrics["ham_test_accuracy"] = float((y == yhat).mean())
        metrics["ham_test_f1_per_class"] = dict(zip(ISIC2019_CLASSES, map(float, per_class_f1(y, yhat, len(ISIC2019_CLASSES)))))
    if arm in ("S", "J"):
        metrics["ham_test_dice"] = float(ham["dice"].mean())
        metrics["ham_test_iou"] = float(ham["iou"].mean())
        isic = run(model, SegTestDataset(args.data_root, image_size=cfg["image_size"]),
                   device, args.batch_size, args.num_workers, args.max_batches)
        isic.to_csv(args.run_dir / "test_isic2018_per_image.csv", index=False)
        metrics["isic2018_test_n"] = len(isic)
        metrics["isic2018_test_dice"] = float(isic["dice"].mean())
        metrics["isic2018_test_iou"] = float(isic["iou"].mean())

    (args.run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
