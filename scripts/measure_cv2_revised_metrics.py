"""
Compute B1 and D against the REVISED CV-2 metric definitions
(Section 22 revised): image-level recall, per-image false-candidate
burden, with binary FPR and box-level recall retained as reported
secondaries.

This is characterization of existing checkpoints against new metric
DEFINITIONS -- not target-fitting. Numeric pass/fail targets are set as a
separate, committed decision AFTER seeing this achievable range.

Runs locally on already-generated prediction CSVs. No GPU, no pod.

The prediction CSVs are generated at conf=0.001 (every candidate kept),
so this script applies an operating confidence threshold in postprocessing
and reports the revised metrics at that threshold. Default 0.25 matches
the locked operating point used in prior evaluation.

Usage:
    python scripts/measure_cv2_revised_metrics.py [--conf 0.25]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

PRED_FILES = {
    "B1 (1280)": REPO_ROOT
    / "evaluation/cv2/prediction_diagnostics/b1_1280/predictions.csv",
    "D (oversampled)": REPO_ROOT
    / "evaluation/cv2/prediction_diagnostics/d1_sun_damage_oversample/predictions.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure CV-2 checkpoints against revised metrics."
    )
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def measure(name: str, path: Path, conf: float) -> None:
    if not path.exists():
        print(f"[SKIP] {name}: predictions not found at {path}")
        print(
            "       (D's predictions.csv must be generated on the pod via "
            "analyze_cv2_predictions.py and pulled locally before this "
            "row can be computed.)"
        )
        print()
        return

    df = pd.read_csv(path)

    # Apply operating threshold in postprocessing.
    kept = df[df["confidence"] >= conf].copy()

    print("=" * 70)
    print(f"{name}  (operating conf = {conf})")
    print("=" * 70)

    # ---- Image-level recall (PRIMARY GATE, Change 1) ----
    # Lesion-containing images = those with gt_boxes > 0.
    lesion_images = df[df["gt_boxes"] > 0]["image_id"].unique()
    n_lesion_images = len(lesion_images)

    # An image is "caught" if it has >= 1 kept true-positive candidate.
    kept_tp = kept[kept["matched"] == True]
    caught_images = kept_tp["image_id"].unique()
    caught_lesion_images = np.intersect1d(caught_images, lesion_images)

    image_recall = (
        len(caught_lesion_images) / n_lesion_images
        if n_lesion_images
        else 0.0
    )

    # ---- Box-level recall (reported secondary) ----
    total_gt = int(df["gt_boxes"].groupby(df["image_id"]).first().sum())
    total_matched_kept = int((kept["matched"] == True).sum())
    box_recall = total_matched_kept / total_gt if total_gt else 0.0

    # ---- Zero-lesion false-candidate burden (PRIMARY GATE, Change 2) ----
    zero_ids = df[df["zero_lesion"] == True]["image_id"].unique()
    n_zero = len(zero_ids)

    kept_zero = kept[kept["zero_lesion"] == True]
    # false candidates per zero-lesion image (unmatched by definition on
    # zero-lesion images, since there are no GT boxes to match)
    per_image_fp = (
        kept_zero.groupby("image_id").size()
        .reindex(zero_ids, fill_value=0)
    )

    burden_median = float(per_image_fp.median())
    burden_p90 = float(per_image_fp.quantile(0.90))
    burden_max = int(per_image_fp.max())

    # ---- Binary zero-lesion FPR (reported secondary) ----
    binary_fpr = float((per_image_fp > 0).mean())

    # ---- Dense / high-sun-damage burden stratum (reported) ----
    dense_ids = df[df["density_bucket"] == "10+"]["image_id"].unique()
    # note: dense zero-lesion images don't exist (10+ implies lesions);
    # dense stratum FP burden is about false EXTRA candidates on images
    # that do contain lesions -- report unmatched kept candidates per
    # dense image as a proxy for over-prediction in dense scenes.
    kept_dense = kept[kept["density_bucket"] == "10+"]
    dense_fp_per_image = (
        kept_dense[kept_dense["matched"] == False]
        .groupby("image_id").size()
        .reindex(dense_ids, fill_value=0)
    )
    dense_fp_median = (
        float(dense_fp_per_image.median()) if len(dense_ids) else 0.0
    )

    print(f"  Lesion-containing images:        {n_lesion_images}")
    print(f"  Zero-lesion images:              {n_zero}")
    print()
    print("  PRIMARY GATES (revised):")
    print(f"    Image-level recall:            {image_recall:.4f}")
    print(f"    Zero-lesion FP burden median:  {burden_median:.2f}")
    print(f"    Zero-lesion FP burden p90:     {burden_p90:.2f}")
    print()
    print("  REPORTED SECONDARIES:")
    print(f"    Box-level recall:              {box_recall:.4f}")
    print(f"    Binary zero-lesion FPR:        {binary_fpr:.4f}")
    print(f"    Zero-lesion FP burden max:     {burden_max}")
    print(f"    Dense-scene extra-cand median: {dense_fp_median:.2f}")
    print()


def main() -> None:
    args = parse_args()
    print()
    print(
        "CV-2 REVISED METRIC CHARACTERIZATION "
        "(not target-fitting -- see Section 22 revised)"
    )
    print()
    for name, path in PRED_FILES.items():
        measure(name, path, args.conf)


if __name__ == "__main__":
    main()