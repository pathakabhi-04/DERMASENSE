"""
CV-3 domain validation on real iToBoS (TBP) crops.

Question: CV-3 was trained and Dice-measured only on ISIC 2018
(dermoscopic). The CV-2->CV-3 geometry interface experiment
(validate_cv2_cv3_interface.py) already showed CV-3 tolerates
detector-style crop geometry -- but it deliberately held domain constant
(ISIC only, simulated boxes). This script asks the domain question: does
CV-3 produce coherent masks on real CV-2 detections from iToBoS
(wide-field TBP), or does the dermoscopic->TBP domain gap break it?

Why this is a proxy-metric experiment, not a Dice experiment: iToBoS has
bounding boxes only, no segmentation masks, so there is no ground truth
to score against. See docs/cv3_domain_validation_spec.md for the full
pre-committed design and decision rule -- read that before interpreting
output.

Method:
  - Real CV-2 B1 true-positive detections on real iToBoS images
    (evaluation/cv2/prediction_diagnostics/b1_1280/predictions.csv,
    matched==True, zero_lesion==False) -- NOT simulated boxes.
  - Each detection -> src.inference.crop_normalize.crop_and_normalize()
    at the validated margin=0.25 default -> CV-3 -> predicted mask.
  - Proxy metrics (degenerate-mask rate, fg-area fraction, border-touch
    rate) computed on iToBoS predictions AND, as a control, on an
    ISIC-test-set run through the identical crop+predict code path (GT
    box + margin=0.25, no ISIC-specific ground truth used here -- masks
    are not needed for these proxies).
  - A stratified (by sun_damage_level) random sample of n=50 images is
    written out as a contact sheet + a ratings CSV for manual visual
    audit. That audit, not the proxy metrics, is the actual decision
    signal -- see the spec for why.

This is ONE bounded run: proxy metrics computed once (not swept), one
audit sample (not re-drawn to chase a rounder number). Apply the
decision rule in docs/cv3_domain_validation_spec.md and stop.

Usage (local, CPU, no pod needed):
    python -m scripts.validate_cv3_domain_itobos \
        --weights checkpoints/cv3_512/best.pt \
        --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.inference.crop_normalize import (
    crop_and_normalize,
    pixel_box_to_norm,
    CV3_INPUT_SIZE,
)
from src.segmentation.inference import load_segmentation_model, predict_mask
from scripts.validate_cv2_cv3_interface import tight_box_from_mask

REPO_ROOT = Path(__file__).resolve().parents[1]

ITOBOS_IMAGES_DIR = REPO_ROOT / "data/raw/itobos/_train/_train/images"
ITOBOS_METADATA = REPO_ROOT / "data/raw/itobos/_train/_train/metadata.csv"
CV2_PREDICTIONS = (
    REPO_ROOT / "evaluation/cv2/prediction_diagnostics/b1_1280/predictions.csv"
)
ISIC_TEST_SPLIT = REPO_ROOT / "data/splits/isic2018_task1/test.csv"

AUDIT_SAMPLE_SIZE = 50
RANDOM_SEED = 42
DEGENERATE_FG_FRAC_HIGH = 0.95

# CV-3 (UNet) forward pass on this CPU measures ~0.63s/crop (profiled
# before this run). All 5686 iToBoS TP crops would be ~60 minutes for a
# proxy metric that is explicitly a sanity net, not the decision signal
# (the visual audit is). A fixed random subsample gives the same
# precision that matters here -- at n=1000 the standard error on a rate
# estimate is ~1.5%, far tighter than needed to catch the "qualitative
# shift" the spec's decision rule looks for -- so default to bounded
# effort rather than exhaustive. Override with --max-itobos-crops if a
# full run is ever specifically warranted.
DEFAULT_MAX_ITOBOS_CROPS = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CV-3 domain validation on real iToBoS TBP crops via proxy "
            "metrics + visual audit sample. See "
            "docs/cv3_domain_validation_spec.md."
        )
    )
    p.add_argument(
        "--weights", type=Path, default=REPO_ROOT / "checkpoints/cv3_512/best.pt"
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--margin",
        type=float,
        default=0.25,
        help="crop margin -- matches the validated CV-2->CV-3 interface default",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "analysis/quality/cv3_domain_itobos",
    )
    p.add_argument(
        "--max-itobos-crops",
        type=int,
        default=DEFAULT_MAX_ITOBOS_CROPS,
        help=(
            "cap on iToBoS TP crops for the proxy-metric pass (a fixed "
            "random subsample, seeded) -- this is a sanity-net metric, "
            "not the decision signal, so full exhaustiveness isn't "
            "needed. Pass 0 to run all available crops."
        ),
    )
    return p.parse_args()


# load_model / predict_mask / pixel_box_to_norm now live in src/ so the
# inference pipeline and this script share one definition:
#   src.segmentation.inference.{load_segmentation_model, predict_mask}
#   src.inference.crop_normalize.pixel_box_to_norm
load_model = load_segmentation_model


def proxy_metrics_for_mask(mask: np.ndarray) -> dict:
    h, w = mask.shape
    fg = float(mask.sum())
    fg_frac = fg / (h * w)
    degenerate = fg == 0 or fg_frac > DEGENERATE_FG_FRAC_HIGH
    border = np.concatenate(
        [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]
    )
    border_touch = bool(border.sum() > 0)
    return {
        "fg_frac": fg_frac,
        "degenerate": bool(degenerate),
        "border_touch": border_touch,
    }


def run_itobos(
    model, device: torch.device, margin: float, max_crops: int = 0
) -> pd.DataFrame:
    preds = pd.read_csv(CV2_PREDICTIONS)
    tp = preds[(preds["matched"] == True) & (preds["zero_lesion"] == False)]  # noqa: E712

    if max_crops and len(tp) > max_crops:
        tp = tp.sample(n=max_crops, random_state=RANDOM_SEED)

    records = []
    for image_id, group in tp.groupby("image_id"):
        img_path = ITOBOS_IMAGES_DIR / f"{image_id}.png"
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        img_h, img_w = image_bgr.shape[:2]

        for _, row in group.iterrows():
            box_norm = pixel_box_to_norm(
                row["x1"], row["y1"], row["x2"], row["y2"], img_w, img_h
            )
            tensor, px_box = crop_and_normalize(image_bgr, box_norm, margin=margin)
            mask = predict_mask(model, tensor, device)
            metrics = proxy_metrics_for_mask(mask)
            metrics.update(
                {
                    "image_id": image_id,
                    "confidence": row["confidence"],
                    "px_box": px_box,
                }
            )
            records.append(metrics)

    return pd.DataFrame(records)


def run_isic_control(model, device: torch.device, margin: float) -> pd.DataFrame:
    df = pd.read_csv(ISIC_TEST_SPLIT)
    df = df[df["cv3_eligible"].astype(bool)]

    records = []
    for _, row in df.iterrows():
        img_path = REPO_ROOT / row["image_path"]
        mask_path = REPO_ROOT / row["mask_path"]
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or gt_mask is None:
            continue

        box = tight_box_from_mask(gt_mask)
        if box is None:
            continue

        tensor, _ = crop_and_normalize(image_bgr, box, margin=margin)
        mask = predict_mask(model, tensor, device)
        metrics = proxy_metrics_for_mask(mask)
        metrics["image_id"] = row.get("image_id", img_path.stem)
        records.append(metrics)

    return pd.DataFrame(records)


def draw_overlay(image_bgr: np.ndarray, px_box, mask: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = px_box
    crop = image_bgr[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, CV3_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    overlay = crop_resized.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def build_audit_sample(
    model, device: torch.device, margin: float, sample_size: int, seed: int
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    preds = pd.read_csv(CV2_PREDICTIONS)
    tp = preds[(preds["matched"] == True) & (preds["zero_lesion"] == False)]  # noqa: E712
    # one representative detection per image (highest confidence) so the
    # audit samples images, not multi-lesion images repeatedly
    best = tp.sort_values("confidence", ascending=False).drop_duplicates(
        "image_id", keep="first"
    )

    meta = pd.read_csv(ITOBOS_METADATA)[["image_id", "sun_damage_level", "body_part"]]
    pool = best.merge(meta, on="image_id", how="inner")

    strata = pool.groupby("sun_damage_level")
    n_total = len(pool)
    sampled_parts = []
    for level, group in strata:
        n_stratum = max(1, round(len(group) / n_total * sample_size))
        sampled_parts.append(
            group.sample(n=min(n_stratum, len(group)), random_state=seed)
        )
    sample = pd.concat(sampled_parts).sample(frac=1.0, random_state=seed)
    sample = sample.head(sample_size).sort_values(
        ["sun_damage_level", "image_id"]
    ).reset_index(drop=True)

    thumbnails = []
    rows_out = []
    for _, row in sample.iterrows():
        image_id = row["image_id"]
        img_path = ITOBOS_IMAGES_DIR / f"{image_id}.png"
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        img_h, img_w = image_bgr.shape[:2]
        box_norm = pixel_box_to_norm(
            row["x1"], row["y1"], row["x2"], row["y2"], img_w, img_h
        )
        tensor, px_box = crop_and_normalize(image_bgr, box_norm, margin=margin)
        mask = predict_mask(model, tensor, device)
        thumbnails.append(draw_overlay(image_bgr, px_box, mask))
        rows_out.append(
            {
                "image_id": image_id,
                "sun_damage_level": row["sun_damage_level"],
                "body_part": row["body_part"],
                "confidence": row["confidence"],
                "rating": "",  # fill in manually: reasonable | fail
            }
        )

    return pd.DataFrame(rows_out), thumbnails


def save_contact_sheet(thumbnails: list[np.ndarray], out_path: Path, cols: int = 10) -> None:
    if not thumbnails:
        return
    rows = -(-len(thumbnails) // cols)  # ceil
    tile_h, tile_w = thumbnails[0].shape[:2]
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 255, dtype=np.uint8)
    for i, thumb in enumerate(thumbnails):
        r, c = divmod(i, cols)
        sheet[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = thumb
    cv2.imwrite(str(out_path), sheet)


def summarize(itobos_df: pd.DataFrame, isic_df: pd.DataFrame) -> str:
    lines = [
        "CV-3 Domain Validation on iToBoS (TBP) Crops -- Proxy Metrics",
        "=" * 60,
        "",
        f"iToBoS TP crops: {len(itobos_df)} (from "
        f"{itobos_df['image_id'].nunique()} images)",
        f"ISIC control crops: {len(isic_df)}",
        "",
        f"{'metric':<28}{'iToBoS':>12}{'ISIC control':>16}",
        f"{'degenerate-mask rate':<28}"
        f"{itobos_df['degenerate'].mean():>12.3f}"
        f"{isic_df['degenerate'].mean():>16.3f}",
        f"{'border-touch rate':<28}"
        f"{itobos_df['border_touch'].mean():>12.3f}"
        f"{isic_df['border_touch'].mean():>16.3f}",
        f"{'fg_frac median':<28}"
        f"{itobos_df['fg_frac'].median():>12.3f}"
        f"{isic_df['fg_frac'].median():>16.3f}",
        f"{'fg_frac IQR':<28}"
        f"{itobos_df['fg_frac'].quantile(0.75) - itobos_df['fg_frac'].quantile(0.25):>12.3f}"
        f"{isic_df['fg_frac'].quantile(0.75) - isic_df['fg_frac'].quantile(0.25):>16.3f}",
        "",
        "These proxies are a sanity net, NOT the decision signal. Compare "
        "against the pre-committed criteria in "
        "docs/cv3_domain_validation_spec.md, and treat the visual audit "
        "sample (audit_sample.csv + audit_contact_sheet.jpg) as the real "
        "decision signal: rate each row 'reasonable' or 'fail' by eye, "
        "then apply the >=80% gate.",
        "",
        "Do not sweep additional proxy metrics or enlarge the sample -- "
        "one run, one decision, per the spec's anti-rabbit-hole clause.",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights, device)

    print(
        f"Running CV-3 on real iToBoS TP crops "
        f"(cap={args.max_itobos_crops or 'none'})..."
    )
    itobos_df = run_itobos(model, device, args.margin, args.max_itobos_crops)
    itobos_df.drop(columns=["px_box"]).to_csv(
        args.output / "itobos_proxy_metrics.csv", index=False
    )

    print("Running CV-3 on ISIC control crops (same geometry, GT box)...")
    isic_df = run_isic_control(model, device, args.margin)
    isic_df.to_csv(args.output / "isic_control_proxy_metrics.csv", index=False)

    print("Building stratified visual-audit sample...")
    audit_df, thumbnails = build_audit_sample(
        model, device, args.margin, AUDIT_SAMPLE_SIZE, RANDOM_SEED
    )
    audit_df.to_csv(args.output / "audit_sample.csv", index=False)
    save_contact_sheet(thumbnails, args.output / "audit_contact_sheet.jpg")

    summary_text = summarize(itobos_df, isic_df)
    (args.output / "summary.txt").write_text(summary_text + "\n")

    print()
    print(summary_text)
    print()
    print(f"Output written to: {args.output}")
    print(
        f"NEXT STEP (manual): open {args.output / 'audit_contact_sheet.jpg'}, "
        f"rate each of the {len(audit_df)} crops in "
        f"{args.output / 'audit_sample.csv'} as 'reasonable' or 'fail', "
        f"then apply the >=80% gate from docs/cv3_domain_validation_spec.md."
    )


if __name__ == "__main__":
    main()
