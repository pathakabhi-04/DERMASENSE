"""Collect Experiment 3 and apply docs/academic_vit_spec.md §5 mechanically.

    PYTHONPATH=. python3 scripts/academic_vit/collect.py \
        --run-root runs/academic_vit --out-dir runs/academic_vit/results \
        --r50b evaluation/academic_vit/r50_on_bundle.json \
        --r50-cv3-per-image runs/academic_r50/S_seed42/test_isic2018_per_image.csv \
        --r50-attention evaluation/academic_r50/attention_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

MARGIN_CLS = 0.02
MARGIN_SEG = 0.01
R50_CV3_DICE = 0.8964


def bootstrap(base: pd.DataFrame, cand: pd.DataFrame, n: int = 10_000, seed: int = 42) -> dict:
    m = base.merge(cand, on="image_id", suffixes=("_base", "_cand"))
    d = (m.dice_cand - m.dice_base).to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(m), "mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "vit_better": int((d > 0).sum()), "r50_better": int((d < 0).sum())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--r50b", type=Path, required=True)
    p.add_argument("--r50-cv3-per-image", type=Path, required=True)
    p.add_argument("--r50-attention", type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    r50b = json.loads(a.r50b.read_text())
    vit4 = json.loads((a.run_root / "cv4/test_metrics.json").read_text())
    vit3 = json.loads((a.run_root / "S_seed42/test_metrics.json").read_text())

    cv4_match = vit4["macro_f1"] >= r50b["macro_f1"] - MARGIN_CLS
    cv4_beats = vit4["macro_f1"] >= r50b["macro_f1"] + MARGIN_CLS
    cv3_match = vit3["dice_mean"] >= R50_CV3_DICE - MARGIN_SEG
    cv3_beats = vit3["dice_mean"] >= R50_CV3_DICE + MARGIN_SEG
    outcome = {2: "A", 1: "B", 0: "C"}[int(cv4_match) + int(cv3_match)]
    boot = bootstrap(pd.read_csv(a.r50_cv3_per_image),
                     pd.read_csv(a.run_root / "S_seed42/test_isic2018_per_image.csv"))

    perf = pd.DataFrame([
        {"task": "CV-4", "metric": "test_macro_f1", "vit": vit4["macro_f1"], "resnet50": r50b["macro_f1"],
         "match_at": r50b["macro_f1"] - MARGIN_CLS, "beats_at": r50b["macro_f1"] + MARGIN_CLS,
         "matches": cv4_match, "beats": cv4_beats},
        {"task": "CV-4", "metric": "mel_recall (secondary)", "vit": vit4["mel_recall"], "resnet50": r50b["mel_recall"],
         "vit_wilson95": vit4["mel_recall_wilson95"], "resnet50_wilson95": r50b["mel_recall_wilson95"]},
        {"task": "CV-4", "metric": "accuracy (secondary)", "vit": vit4["accuracy"], "resnet50": r50b["accuracy"]},
        {"task": "CV-3", "metric": "test_dice", "vit": vit3["dice_mean"], "resnet50": R50_CV3_DICE,
         "match_at": R50_CV3_DICE - MARGIN_SEG, "beats_at": R50_CV3_DICE + MARGIN_SEG,
         "matches": cv3_match, "beats": cv3_beats},
        {"task": "CV-3", "metric": "paired bootstrap ViT - R50 Dice (secondary)", "vit": boot["mean_diff"],
         "vit_wilson95": boot["ci95"]},
        {"task": "CV-3", "metric": "test_iou (secondary)", "vit": vit3["iou_mean"], "resnet50": 0.8295},
    ])
    perf.to_csv(a.out_dir / "performance.csv", index=False)

    att = []
    for name in ("cv4_rollout", "cv4_gradcam", "cv3_rollout", "cv3_gradcam"):
        m = json.loads((a.run_root / f"attention/{name}/metrics.json").read_text())
        att.append({"model": "ViT-B/16", **{k: v for k, v in m.items() if k != "empty_prediction_ids"}})
        shutil.copyfile(a.run_root / f"attention/{name}/attention_{name}.jpg", a.out_dir / f"attention_{name}.jpg")
    r50 = pd.read_csv(a.r50_attention)
    for _, r in r50[r50.task.isin(["cv3", "cv4"])].iterrows():
        att.append({"model": "ResNet-50 (Experiment 2)", **{k: v for k, v in r.items() if pd.notna(v)},
                    "task": f"{r.task}_gradcam"})
    pd.DataFrame(att).to_csv(a.out_dir / "attention_metrics.csv", index=False)

    decision = {"cv4_matches": cv4_match, "cv4_beats": cv4_beats, "cv3_matches": cv3_match, "cv3_beats": cv3_beats,
                "outcome": outcome, "r50b_macro_f1": r50b["macro_f1"], "cv3_bootstrap": boot,
                "claims": {r["task"]: r.get("claim_allowed") for r in att if r["model"] == "ViT-B/16"}}
    (a.out_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    print(perf.to_string(index=False))
    print(pd.DataFrame(att)[["model", "task", "n", "pointing_game", "energy_in_lesion", "sanity_median_spearman",
                             "sanity_pass", "claim_allowed"]].to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
