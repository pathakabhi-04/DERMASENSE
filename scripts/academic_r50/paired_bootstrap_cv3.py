"""Paired bootstrap: ResNet-50 U-Net vs the production CV-3 U-Net, per test image.

The same method as the CV-3 loss ablation on main (docs/cv3_segmentation_baseline.md §7):
10,000 resamples of the 260 per-image differences, 95% percentile interval.

    python3 scripts/academic_r50/paired_bootstrap_cv3.py \
        --baseline ~/dermasense/evaluation/cv3/per_image_metrics.csv \
        --candidate runs/academic_r50/S_seed42/test_isic2018_per_image.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    m = pd.read_csv(a.baseline).merge(pd.read_csv(a.candidate), on="image_id", suffixes=("_base", "_cand"))
    if len(m) != 260:
        raise SystemExit(f"expected 260 paired test images, got {len(m)}")
    rng = np.random.default_rng(a.seed)
    out = {"n": len(m), "resamples": a.resamples, "seed": a.seed}
    for metric in ("dice", "iou"):
        d = (m[f"{metric}_cand"] - m[f"{metric}_base"]).to_numpy()
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.resamples)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[metric] = {"baseline_mean": float(m[f"{metric}_base"].mean()), "candidate_mean": float(m[f"{metric}_cand"].mean()),
                       "mean_diff": float(d.mean()), "median_diff": float(np.median(d)), "ci95": [float(lo), float(hi)],
                       "candidate_better": int((d > 0).sum()), "baseline_better": int((d < 0).sum()), "ties": int((d == 0).sum())}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
