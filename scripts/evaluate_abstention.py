"""
Can capture quality tell us when CV-4 is about to be wrong?

## The harm this targets

On the external set, 37 of 99 assessed melanomas were classified as NEV
or SEK and sent to MONITOR (external_6class_result.md). Those are the
dangerous outcome: the pipeline looked at a melanoma and produced a
non-referral. If the system could instead say "I cannot assess this
image", they become honest non-answers -- which the pipeline already
treats as a first-class outcome (PipelineOutcome.QUALITY_REJECTED /
NO_CANDIDATES) rather than a reassurance.

## Why quality signals rather than model confidence

Model confidence does not know when it is wrong. On those same
melanomas:

    calibrated_confidence   routed 0.7031   missed 0.7028
    confidence              routed 0.7902   missed 0.7729

No separation. Capture quality does separate:

    crop_contrast           routed 0.9352   missed 0.5968
    mask_area_fraction      routed 0.4346   missed 0.3107

So this tests signals computed BEFORE classification (CV-1 quality,
CV-3 mask evidence), not the classifier's own output.

## Pre-committed method

Choosing a threshold and scoring it on the same images is fitting to the
test set. So: the external set is split 50/50, stratified by
(true_class, was_the_melanoma_missed), fixed seed. Each threshold is
chosen on **half A** and reported on **half B**, untouched.

Single-signal thresholds only. Fitting a multivariate rule on 37
positives would overfit, and refusing to is the point of the
pre-commitment.

  Primary metric  fraction of DANGEROUS MISSES (melanoma -> NEV/SEK)
                  that the rule abstains on, measured on half B
  Cost metric     overall abstention rate on half B, and the abstention
                  rate among correctly-handled benign lesions
  Null            random abstention at the same rate, which catches
                  misses in proportion to how much it abstains. A rule
                  that does not beat this is doing nothing.

  Decision rule (fixed before running):
    A signal is USEFUL if, on half B, it abstains >= 50% of dangerous
    misses while abstaining <= 25% of all images.
    If no signal clears that, quality-keyed abstention is not the answer
    and this line closes.

    PYTHONPATH=. python3 scripts/evaluate_abstention.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.training.metrics import wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = REPO_ROOT / "analysis/product_eval/cv1_cv4_assembly/external_predictions.csv"
OUT = REPO_ROOT / "analysis/quality/mel_sensitivity/abstention_result.csv"

BENIGN_PREDICTIONS = {"NEV", "SEK"}
# Computed before CV-4 runs: CV-1 quality and CV-3 mask evidence.
SIGNALS = ("quality_score", "crop_contrast", "crop_blur", "mask_area_fraction")
SEED = 20260915
MAX_ABSTENTION = 0.25
MIN_MISSES_CAUGHT = 0.50


def load() -> pd.DataFrame:
    data = pd.read_csv(PREDICTIONS)
    data = data[data["outcome"] == "ASSESSED"].drop_duplicates("image_id").copy()
    data = data.dropna(subset=["predicted_class"])
    data["routed"] = ~data["predicted_class"].isin(BENIGN_PREDICTIONS)
    data["is_melanoma"] = data["true_class"] == "MEL"
    # The harm: a melanoma the pipeline assessed and did NOT refer.
    data["dangerous_miss"] = data["is_melanoma"] & ~data["routed"]
    # Correctly handled benign: not referred, and genuinely benign.
    data["benign_ok"] = data["true_class"].isin(BENIGN_PREDICTIONS) & ~data["routed"]
    return data


def split_halves(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    strata = data["true_class"].astype(str) + "_" + data["dangerous_miss"].astype(str)
    in_a = np.zeros(len(data), dtype=bool)
    for _, index in data.groupby(strata).groups.items():
        positions = data.index.get_indexer(index)
        chosen = rng.permutation(len(positions))[: len(positions) // 2]
        in_a[positions[chosen]] = True
    return data[in_a], data[~in_a]


def sweep(frame: pd.DataFrame, signal: str) -> list[dict]:
    """Abstain when the signal is BELOW a threshold (low quality)."""
    values = frame[signal].dropna()
    results = []
    for threshold in np.quantile(values, np.linspace(0.02, 0.60, 40)):
        abstain = frame[signal] < threshold
        misses = frame["dangerous_miss"]
        if misses.sum() == 0:
            continue
        results.append({
            "signal": signal,
            "threshold": float(threshold),
            "abstention_rate": float(abstain.mean()),
            "misses_caught": float(abstain[misses].mean()),
            "benign_ok_abstained": float(abstain[frame["benign_ok"]].mean())
            if frame["benign_ok"].any() else float("nan"),
        })
    return results


def main() -> None:
    data = load()
    half_a, half_b = split_halves(data)
    print(f"{len(data)} assessed images, {int(data['dangerous_miss'].sum())} dangerous misses")
    print(f"  half A (choose threshold): {len(half_a)}, "
          f"{int(half_a['dangerous_miss'].sum())} misses")
    print(f"  half B (report only)     : {len(half_b)}, "
          f"{int(half_b['dangerous_miss'].sum())} misses\n")

    rows = []
    print(f"{'signal':20}{'thr':>8}{'A:absta':>9}{'A:caught':>10}"
          f"{'B:absta':>9}{'B:caught':>10}{'B:null':>8}{'verdict':>12}")
    print("-" * 86)

    for signal in SIGNALS:
        if signal not in data.columns or data[signal].isna().all():
            print(f"{signal:20}  (absent)")
            continue

        # Choose on A: most misses caught, subject to the abstention budget.
        candidates = [r for r in sweep(half_a, signal)
                      if r["abstention_rate"] <= MAX_ABSTENTION]
        if not candidates:
            print(f"{signal:20}  no operating point within the {MAX_ABSTENTION:.0%} budget on A")
            continue
        best = max(candidates, key=lambda r: r["misses_caught"])

        # Apply that exact threshold to B, untouched.
        abstain_b = half_b[signal] < best["threshold"]
        misses_b = half_b["dangerous_miss"]
        caught_b = float(abstain_b[misses_b].mean()) if misses_b.any() else float("nan")
        rate_b = float(abstain_b.mean())
        # Null: abstaining at random at the same rate catches misses in
        # proportion to the rate. A rule must beat this to mean anything.
        null_b = rate_b

        useful = caught_b >= MIN_MISSES_CAUGHT and rate_b <= MAX_ABSTENTION
        verdict = "USEFUL" if useful else ("no lift" if caught_b <= null_b else "weak")
        print(f"{signal:20}{best['threshold']:>8.3f}{best['abstention_rate']:>9.2%}"
              f"{best['misses_caught']:>10.2%}{rate_b:>9.2%}{caught_b:>10.2%}"
              f"{null_b:>8.2%}{verdict:>12}")

        low, high = wilson_interval(int(abstain_b[misses_b].sum()), int(misses_b.sum()))
        rows.append({
            "signal": signal, "threshold": best["threshold"],
            "a_abstention": best["abstention_rate"], "a_caught": best["misses_caught"],
            "b_abstention": rate_b, "b_caught": caught_b, "b_null": null_b,
            "b_caught_ci_lo": low, "b_caught_ci_hi": high,
            "b_benign_ok_abstained": float(abstain_b[half_b["benign_ok"]].mean())
            if half_b["benign_ok"].any() else float("nan"),
            "useful": useful,
        })

    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT, index=False)
        print(f"\n-> {OUT}")
        best_row = frame.loc[frame["b_caught"].idxmax()]
        print(f"\nbest on held-out half B: {best_row['signal']} caught "
              f"{best_row['b_caught']:.1%} of dangerous misses "
              f"[{best_row['b_caught_ci_lo']:.2f}, {best_row['b_caught_ci_hi']:.2f}] "
              f"at {best_row['b_abstention']:.1%} abstention "
              f"(null {best_row['b_null']:.1%}), "
              f"costing {best_row['b_benign_ok_abstained']:.1%} of correctly-handled benign")

    print("\nDecision rule: USEFUL requires >=50% of dangerous misses caught on B "
          f"at <=25% abstention.")
    if not any(r["useful"] for r in rows):
        print("NO SIGNAL CLEARS IT -- quality-keyed abstention is not the answer.")


if __name__ == "__main__":
    main()
