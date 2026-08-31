"""
CV-2 -> CV-3 interface validation via crop-geometry perturbation.

Question: CV-3 (segmentation) was trained on whole ISIC images
squash-resized to 512. In the real pipeline it will instead receive
CROPS produced by CV-2 detection. Does CV-3 degrade when the input is a
tighter / more off-center crop than the full lesion-centric frames it
trained on?

Method (isolates crop GEOMETRY, holds domain constant):
  - Use ISIC 2018 test set (260 images, all dermoscopic, WITH masks).
  - For each image, derive the tight lesion bounding box from the
    ground-truth mask.
  - Generate crops across a grid of (margin, center_offset) simulating
    the range CV-2 realistically produces:
      margin large  -> approximates CV-3's full-frame training input
                       (sanity check: should recover ~baseline Dice)
      margin tight  -> approximates an aggressive CV-2 crop (risk case)
  - Run CV-3 on each crop, resized/preprocessed identically to training.
  - Compute Dice against the mask cropped+resized the SAME way.
  - Report Dice vs crop-geometry curve.

Dice is imported from src.segmentation.metrics (identical definition to
the 0.86 baseline) so the comparison is valid.

Ground-truth note: this uses ISIC masks (which exist). It does NOT use
CV-2's real iToBoS crops, because iToBoS has no segmentation masks -- so
Dice cannot be computed there. This experiment isolates the geometry
axis with ground truth; the domain axis (iToBoS TBP vs ISIC dermoscopic)
is a SEPARATE, later question and is explicitly out of scope here.

Usage:
    python scripts/validate_cv2_cv3_interface.py \
        --weights checkpoints/cv3_512/best.pt \
        --split data/splits/isic2018_task1/test.csv \
        --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.segmentation.model import build_model
from src.segmentation.metrics import segmentation_dice

# reuse the interface transform
from src.inference.crop_normalize import (
    CV3_INPUT_SIZE,
    expand_and_clip_box,
    preprocess_crop_for_cv3,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Perturbation grid. margin = context expansion per side (fraction of box);
# offset = centering error (fraction of box size, applied to x and y).
MARGINS = [1.0, 0.5, 0.25, 0.1, 0.0]
OFFSETS = [0.0, 0.1, 0.2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate CV-2 -> CV-3 interface via crop perturbation."
    )
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument(
        "--split",
        type=Path,
        default=REPO_ROOT / "data/splits/isic2018_task1/test.csv",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "analysis/quality/cv2_cv3_interface",
    )
    return p.parse_args()


def tight_box_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return normalized (xc, yc, w, h) tight box around mask foreground."""
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    h, w = mask.shape[:2]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bw = x2 - x1
    bh = y2 - y1
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    return (xc / w, yc / h, bw / w, bh / h)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    args.output.mkdir(parents=True, exist_ok=True)

    model = build_model().to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv(args.split)
    df = df[df["cv3_eligible"].astype(bool)]

    height, width = CV3_INPUT_SIZE

    records = []

    with torch.no_grad():
        for margin in MARGINS:
            for offset in OFFSETS:
                dice_scores = []
                skipped = 0

                for _, row in df.iterrows():
                    img_path = REPO_ROOT / row["image_path"]
                    mask_path = REPO_ROOT / row["mask_path"]

                    if not img_path.exists() or not mask_path.exists():
                        skipped += 1
                        continue

                    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if image_bgr is None or mask is None:
                        skipped += 1
                        continue

                    box = tight_box_from_mask(mask)
                    if box is None:
                        skipped += 1
                        continue

                    img_h, img_w = image_bgr.shape[:2]
                    # tight box -> pixel xyxy
                    cx, cy, bw, bh = box
                    x1 = (cx - bw / 2) * img_w
                    y1 = (cy - bh / 2) * img_h
                    x2 = (cx + bw / 2) * img_w
                    y2 = (cy + bh / 2) * img_h

                    px = expand_and_clip_box(
                        x1, y1, x2, y2, margin, img_w, img_h,
                        (offset, offset),
                    )
                    a, b, c, d = px

                    crop_bgr = image_bgr[b:d, a:c]
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    inp = preprocess_crop_for_cv3(crop_rgb).to(device)

                    # crop the mask the SAME way, resize to 512 nearest
                    crop_mask = mask[b:d, a:c]
                    crop_mask = cv2.resize(
                        crop_mask, (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    tgt = (crop_mask > 127).astype(np.float32)
                    tgt = torch.from_numpy(tgt).view(1, 1, height, width).to(device)

                    logits = model(inp)
                    dice = segmentation_dice(logits, tgt)
                    dice_scores.append(float(dice))

                mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
                records.append({
                    "margin": margin,
                    "center_offset": offset,
                    "n_images": len(dice_scores),
                    "skipped": skipped,
                    "mean_dice": round(mean_dice, 4),
                })
                print(
                    f"margin={margin:<4} offset={offset:<4} "
                    f"n={len(dice_scores):<4} mean_dice={mean_dice:.4f}"
                )

    out_df = pd.DataFrame(records)
    out_path = args.output / "interface_dice_grid.csv"
    out_df.to_csv(out_path, index=False)

    print()
    print(f"Grid saved to: {out_path}")
    print()
    print("INTERPRETATION GUIDE:")
    print("  - Large margin (1.0) ~ CV-3's full-frame training input.")
    print("    Should recover close to the ~0.86 baseline. If it does NOT,")
    print("    the harness itself is suspect -- investigate before trusting")
    print("    the tight-margin rows.")
    print("  - Tight margin (0.0) + offset ~ aggressive CV-2 crop.")
    print("    This is the interface risk case.")
    print("  - Compare against the pre-committed Dice floor in the spec")
    print("    BEFORE deciding pass/fail.")


if __name__ == "__main__":
    main()