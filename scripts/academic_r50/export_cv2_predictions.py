"""Export Faster R-CNN predictions on iToBoS val in the B1/E predictions.csv schema.

Every candidate at score >= 0.001 is kept (spec Section 4), matched to GT at
IoU >= 0.5 with the greedy IoU-first policy of analyze_cv2_predictions.py.
Writes <run-dir>/predictions.csv (one row per candidate, the B1/E schema) and
<run-dir>/per_image.csv (one row per val image, including images without any
candidate).

    PYTHONPATH=. python3 scripts/academic_r50/export_cv2_predictions.py \
        --data-root /workspace/dermasense_academic/data/academic_r50_bundle \
        --run-dir   /workspace/dermasense_academic/runs/academic_r50/cv2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.academic.detection import ITobosDataset, build_detector, collate, density_bucket, match_predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default="final.pt")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=None, help="Smoke tests only.")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detector(pretrained=False, score_thresh=0.001)
    model.load_state_dict(torch.load(args.run_dir / args.checkpoint, map_location="cpu", weights_only=False)["model"])
    model.to(device).eval()

    ds = ITobosDataset(args.data_root, "val", train=False)
    if args.max_batches is not None:
        ds = Subset(ds, range(min(len(ds), args.max_batches * args.batch_size)))
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)

    preds, images = [], []
    for batch_images, targets, ids in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model([im.to(device) for im in batch_images])
        for out, tgt, image_id in zip(outputs, targets, ids):
            boxes = out["boxes"].float().cpu().numpy()
            scores = out["scores"].float().cpu().numpy()
            gt = tgt["boxes"].numpy()
            matched = match_predictions(gt, boxes, scores)
            n_gt = len(gt)
            for k in range(len(boxes)):
                preds.append({"image_id": image_id, "density_bucket": density_bucket(n_gt), "gt_boxes": n_gt,
                              "prediction_index": k, "confidence": float(scores[k]),
                              "x1": float(boxes[k, 0]), "y1": float(boxes[k, 1]),
                              "x2": float(boxes[k, 2]), "y2": float(boxes[k, 3]),
                              "matched": k in matched, "zero_lesion": n_gt == 0})
            images.append({"image_id": image_id, "gt_boxes": n_gt, "pred_boxes": len(boxes),
                           "matched_boxes": len(matched), "missed_boxes": n_gt - len(matched),
                           "density_bucket": density_bucket(n_gt), "zero_lesion": n_gt == 0})

    pd.DataFrame(preds).to_csv(args.run_dir / "predictions.csv", index=False)
    pd.DataFrame(images).to_csv(args.run_dir / "per_image.csv", index=False)
    print(f"{len(images)} images, {len(preds)} candidates -> {args.run_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
