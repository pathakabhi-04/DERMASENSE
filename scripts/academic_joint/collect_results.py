"""Collect the 9 runs into results.csv and apply the Section 8 decision rule.

Writes <out-dir>/results.csv (one row per arm x seed), <out-dir>/paired.csv
(J - C and J - S per seed) and <out-dir>/decision.json. The rule is applied
mechanically, exactly as committed in the spec; nothing here is tunable.

    PYTHONPATH=. python3 scripts/academic_joint/collect_results.py \
        --run-root runs/academic_joint --out-dir evaluation/academic_joint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SEEDS = (42, 43, 44)
ARMS = ("S", "C", "J")
CLS_MARGIN = -0.02
SEG_MARGIN = -0.01
IMPROVE_MIN = 0.02


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def load_row(run: Path, arm: str, seed: int) -> dict:
    row = {"arm": arm, "seed": seed}
    t = json.loads((run / "test_metrics.json").read_text())
    row["best_epoch"] = t["best_epoch"]
    for k in ("ham_test_macro_f1", "ham_test_accuracy", "ham_test_dice", "ham_test_iou",
              "isic2018_test_dice", "isic2018_test_iou"):
        row[k] = t.get(k)
    for c, v in t.get("ham_test_f1_per_class", {}).items():
        row[f"ham_test_f1_{c}"] = v
    pad = run / "pad_c1_metrics.json"
    if pad.exists():
        p = json.loads(pad.read_text())
        row["pad_test_macro_f1"] = p["pad_test_macro_f1"]
        row["pad_val_macro_f1"] = p["pad_val_macro_f1"]
        row["pad_best_epoch"] = p["best_epoch"]
    return row


def improves(deltas: pd.Series) -> bool:
    return bool((deltas > 0).all() and deltas.mean() >= IMPROVE_MIN)


def main() -> None:
    args = parse_args()
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            run = args.run_root / f"{arm}_seed{seed}"
            if not (run / "test_metrics.json").exists():
                raise SystemExit(f"missing {run}/test_metrics.json -- all 9 runs are required")
            rows.append(load_row(run, arm, seed))
    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "results.csv", index=False)

    by = df.set_index(["arm", "seed"])
    paired = pd.DataFrame({
        "seed": SEEDS,
        "J_minus_C_macro_f1": [by.loc[("J", s), "ham_test_macro_f1"] - by.loc[("C", s), "ham_test_macro_f1"] for s in SEEDS],
        "J_minus_S_dice": [by.loc[("J", s), "ham_test_dice"] - by.loc[("S", s), "ham_test_dice"] for s in SEEDS],
        "J_minus_S_isic2018_dice": [by.loc[("J", s), "isic2018_test_dice"] - by.loc[("S", s), "isic2018_test_dice"] for s in SEEDS],
        "J_minus_C_pad_macro_f1": [by.loc[("J", s), "pad_test_macro_f1"] - by.loc[("C", s), "pad_test_macro_f1"] for s in SEEDS],
    })
    paired.to_csv(args.out_dir / "paired.csv", index=False)

    d_cls, d_seg, d_pad = paired["J_minus_C_macro_f1"], paired["J_minus_S_dice"], paired["J_minus_C_pad_macro_f1"]
    matches = bool(d_cls.mean() >= CLS_MARGIN and d_seg.mean() >= SEG_MARGIN)
    improves_cls = improves(d_cls)
    outcome = ("A" if improves_cls else "B") if matches else "C"
    decision = {
        "mean_J_minus_C_macro_f1": float(d_cls.mean()),
        "mean_J_minus_S_dice": float(d_seg.mean()),
        "matches": matches,
        "improves_classification": improves_cls,
        "outcome": outcome,
        "pad_transfer_mean_J_minus_C": float(d_pad.mean()),
        "pad_transfer_improves": improves(d_pad),
        "rule": f"match: mean dF1 >= {CLS_MARGIN} and mean dDice >= {SEG_MARGIN}; "
                f"improve: dF1 > 0 in all seeds and mean >= {IMPROVE_MIN}",
    }
    (args.out_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    print(df.to_string(index=False))
    print(paired.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
