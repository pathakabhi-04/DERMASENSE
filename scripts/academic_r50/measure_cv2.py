"""CV-2 revised metrics at the operating threshold, for any predictions.csv.

measure() is measure() from scripts/measure_cv2_revised_metrics.py on main,
unchanged except that it returns its numbers instead of only printing them
(the file on main hardcodes its input paths). It is the decision metric
(spec Section 4).

measure_all_zero() is the secondary, corrected burden (spec Section 3.1): it
counts every zero-lesion val image from the split, including those with no
candidate at all, which predictions.csv omits.

    python3 scripts/academic_r50/measure_cv2.py \
        --predictions runs/academic_r50/cv2/predictions.csv \
        --split-csv data/splits/itobos_detection/val.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def measure(df: pd.DataFrame, conf: float) -> dict:
    kept = df[df["confidence"] >= conf].copy()

    lesion_images = df[df["gt_boxes"] > 0]["image_id"].unique()
    n_lesion_images = len(lesion_images)
    kept_tp = kept[kept["matched"] == True]  # noqa: E712 (as on main)
    caught_images = kept_tp["image_id"].unique()
    caught_lesion_images = np.intersect1d(caught_images, lesion_images)
    image_recall = len(caught_lesion_images) / n_lesion_images if n_lesion_images else 0.0

    total_gt = int(df["gt_boxes"].groupby(df["image_id"]).first().sum())
    total_matched_kept = int((kept["matched"] == True).sum())  # noqa: E712
    box_recall = total_matched_kept / total_gt if total_gt else 0.0

    zero_ids = df[df["zero_lesion"] == True]["image_id"].unique()  # noqa: E712
    kept_zero = kept[kept["zero_lesion"] == True]  # noqa: E712
    per_image_fp = kept_zero.groupby("image_id").size().reindex(zero_ids, fill_value=0)

    return {
        "lesion_images": n_lesion_images,
        "zero_lesion_images": len(zero_ids),
        "image_recall": image_recall,
        "burden_median": float(per_image_fp.median()),
        "burden_p90": float(per_image_fp.quantile(0.90)),
        "burden_max": int(per_image_fp.max()),
        "box_recall": box_recall,
        "binary_zero_fpr": float((per_image_fp > 0).mean()),
    }


def measure_all_zero(df: pd.DataFrame, split: pd.DataFrame, conf: float) -> dict:
    kept = df[(df["confidence"] >= conf) & (df["zero_lesion"] == True)]  # noqa: E712
    zero_ids = split.loc[split["lesion_count"] == 0, "image_id"].unique()
    per_image_fp = kept.groupby("image_id").size().reindex(zero_ids, fill_value=0)
    return {
        "zero_lesion_images_all": len(zero_ids),
        "burden_median_all": float(per_image_fp.median()),
        "burden_p90_all": float(per_image_fp.quantile(0.90)),
        "binary_zero_fpr_all": float((per_image_fp > 0).mean()),
    }


def measure_file(predictions: Path, split_csv: Path, conf: float = 0.25) -> dict:
    df = pd.read_csv(predictions)
    return {**measure(df, conf), **measure_all_zero(df, pd.read_csv(split_csv), conf), "conf": conf}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args()
    print(json.dumps(measure_file(a.predictions, a.split_csv, a.conf), indent=2))


if __name__ == "__main__":
    main()
