"""
Refit CV-4b (the referral head) on ISIC + non-ISIC clinical-photo
features, per the pre-committed plan in
`analysis/quality/mel_sensitivity/cv4b_retrain_scope.md`. Read that file
first -- this script implements it and does not re-litigate any of its
choices (split ratios, label harmonization, decision rule) inline.

Only the *training data* changes; the model stays the same linear probe
as `train_referral_head.py` (logistic regression on frozen features).

    PYTHONPATH=. python3 scripts/refit_referral_head_broad.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from scripts.evaluate_domain_shift import wilson_interval
from scripts.train_referral_head import (
    BENIGN,
    REFER,
    TARGET_BENIGN_REFERRAL,
    TEST_FEATURES,
    TRAIN_FEATURES,
    VAL_FEATURES,
)
from scripts.train_referral_head import load as load_isic

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = REPO_ROOT / "analysis/quality/mel_sensitivity/features"
SPLIT_OUT = REPO_ROOT / "analysis/quality/mel_sensitivity/cv4b_retrain_split.csv"
RESULT_OUT = REPO_ROOT / "analysis/quality/mel_sensitivity/cv4b_retrain_result.json"
HEAD_OUT = Path("checkpoints/referral_head/referral_head.json")

SOURCES = ("ddi", "ddi2", "fitzpatrick17k")
SPLIT_SEED = 20260913  # same seed already used for the Fitzpatrick17k acquisition sample
ISIC_AUC_FLOOR = 0.85
NON_ISIC_AUC_TARGET = 0.80


def load_non_isic() -> pd.DataFrame:
    rows = []
    for source in SOURCES:
        data = np.load(FEATURES_DIR / f"{source}_features.npz", allow_pickle=True)
        n = len(data["is_malignant"])
        rows.append(pd.DataFrame({
            "source": source,
            "is_malignant": data["is_malignant"].astype(bool),
            "is_melanoma": data["is_melanoma"].astype(bool),
            "image_path": data["image_path"],
            "_row": np.arange(n),
        }).assign(_features_file=source))
    combined = pd.concat(rows, ignore_index=True)

    feature_arrays = {s: np.load(FEATURES_DIR / f"{s}_features.npz")["features"] for s in SOURCES}
    features = np.concatenate([feature_arrays[s] for s in SOURCES], axis=0)
    assert features.shape[0] == len(combined)
    return combined, features


def stratified_split(meta: pd.DataFrame) -> pd.Series:
    """60/20/20 by (source, is_malignant), fixed seed. Written to CSV
    before fitting so the assignment is inspectable and fixed."""
    key = meta["source"] + "_" + meta["is_malignant"].astype(str)
    idx = np.arange(len(meta))

    idx_train, idx_rest = train_test_split(
        idx, test_size=0.4, stratify=key, random_state=SPLIT_SEED
    )
    idx_val, idx_test = train_test_split(
        idx_rest, test_size=0.5, stratify=key.iloc[idx_rest], random_state=SPLIT_SEED
    )

    split = pd.Series("train", index=meta.index)
    split.iloc[idx_val] = "val"
    split.iloc[idx_test] = "test"
    return split


def report_source(y_true, y_score, mel_mask, label: str) -> dict:
    auc = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan")
    n_mel = int(mel_mask.sum())
    if n_mel:
        threshold = report_source.threshold
        mel_routed = float((y_score[mel_mask] >= threshold).mean())
        lo, hi = wilson_interval(int((y_score[mel_mask] >= threshold).sum()), n_mel)
    else:
        mel_routed, lo, hi = float("nan"), 0.0, 0.0
    print(f"  {label:<20} n={len(y_true):>5}  AUC={auc:.4f}  "
          f"mel_routed={mel_routed:.4f} [{lo:.2f},{hi:.2f}] (n_mel={n_mel})")
    return {"source": label, "n": len(y_true), "auc": auc, "mel_routed": mel_routed,
            "mel_ci_lo": lo, "mel_ci_hi": hi, "n_mel": n_mel}


def main() -> None:
    print("Loading ISIC features (existing cache)...")
    Xtr_isic, ytr_isic, _ = load_isic("train", TRAIN_FEATURES)
    Xv_isic, yv_isic, dv_isic = load_isic("val", VAL_FEATURES)
    Xt_isic, yt_isic, dt_isic = load_isic("test", TEST_FEATURES)

    print("Loading non-ISIC features (extracted)...")
    meta, X_non_isic = load_non_isic()
    meta["split"] = stratified_split(meta)
    SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    meta.drop(columns=["_row"]).to_csv(SPLIT_OUT, index=False)
    print(f"  split written to {SPLIT_OUT}")
    print(meta.groupby(["source", "split"])["is_malignant"].agg(["size", "sum"]))

    def subset(split_name):
        mask = (meta["split"] == split_name).to_numpy()
        return X_non_isic[mask], meta.loc[mask, "is_malignant"].to_numpy().astype(int), meta.loc[mask]

    Xtr_non, ytr_non, _ = subset("train")
    Xv_non, yv_non, _ = subset("val")
    Xt_non, yt_non, meta_test_non = subset("test")

    Xtr = np.concatenate([Xtr_isic, Xtr_non], axis=0)
    ytr = np.concatenate([ytr_isic, ytr_non], axis=0)
    Xv = np.concatenate([Xv_isic, Xv_non], axis=0)
    yv = np.concatenate([yv_isic, yv_non], axis=0)

    print(f"\nfit: {len(ytr)} (isic {len(ytr_isic)} + non-isic {len(ytr_non)})")
    print(f"threshold: {len(yv)} (isic {len(yv_isic)} + non-isic {len(yv_non)})")

    head = LogisticRegression(max_iter=3000, C=1.0)
    head.fit(Xtr, ytr)

    val_scores = head.predict_proba(Xv)[:, 1]
    threshold = float(np.quantile(val_scores[yv == 0], 1 - TARGET_BENIGN_REFERRAL))
    print(f"\noperating point (val, {TARGET_BENIGN_REFERRAL:.0%} benign-referral budget): {threshold:.4f}")
    report_source.threshold = threshold

    print("\n" + "=" * 72)
    print("HELD-OUT TEST (never used for fitting or threshold selection)")
    print("=" * 72)

    isic_scores = head.predict_proba(Xt_isic)[:, 1]
    isic_result = report_source(yt_isic, isic_scores, dt_isic == "MEL", "ISIC (in-domain)")
    isic_result["benign_referred"] = float((isic_scores[yt_isic == 0] >= threshold).mean())

    non_scores = head.predict_proba(Xt_non)[:, 1]
    non_result = report_source(yt_non, non_scores, meta_test_non["is_melanoma"].to_numpy(),
                                "non-ISIC (combined)")
    non_result["benign_referred"] = float((non_scores[yt_non == 0] >= threshold).mean())
    print(f"    benign referred: ISIC={isic_result['benign_referred']:.4f}  "
          f"non-ISIC={non_result['benign_referred']:.4f}")

    per_source_results = []
    for source in SOURCES:
        rows = meta_test_non["source"] == source
        if rows.sum() == 0:
            continue
        idx = np.where(rows.to_numpy())[0]
        per_source_results.append(
            report_source(yt_non[idx], non_scores[idx],
                           meta_test_non["is_melanoma"].to_numpy()[idx], source)
        )

    isic_ok = isic_result["auc"] >= ISIC_AUC_FLOOR
    non_isic_ok = non_result["auc"] >= NON_ISIC_AUC_TARGET
    success = isic_ok and non_isic_ok

    print("\n" + "=" * 72)
    print(f"decision rule: ISIC test AUC >= {ISIC_AUC_FLOOR} ({'MET' if isic_ok else 'NOT MET'}, "
          f"got {isic_result['auc']:.4f}); "
          f"non-ISIC test AUC >= {NON_ISIC_AUC_TARGET} ({'MET' if non_isic_ok else 'NOT MET'}, "
          f"got {non_result['auc']:.4f})")
    print(f"OVERALL: {'SUCCESS -- refit head will be written' if success else 'NOT MET -- head left unchanged, see result doc'}")
    print("=" * 72)

    RESULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULT_OUT.write_text(json.dumps({
        "split_seed": SPLIT_SEED,
        "threshold": threshold,
        "isic_test": isic_result,
        "non_isic_test_combined": non_result,
        "non_isic_test_per_source": per_source_results,
        "isic_auc_floor": ISIC_AUC_FLOOR,
        "non_isic_auc_target": NON_ISIC_AUC_TARGET,
        "success": success,
    }, indent=2))
    print(f"\nresult -> {RESULT_OUT}")

    if success:
        HEAD_OUT.parent.mkdir(parents=True, exist_ok=True)
        HEAD_OUT.write_text(json.dumps({
            "kind": "logistic_regression",
            "feature_dim": int(Xtr.shape[1]),
            "coef": head.coef_[0].tolist(),
            "intercept": float(head.intercept_[0]),
            "threshold": threshold,
            "target_benign_referral": TARGET_BENIGN_REFERRAL,
            "refer_classes": list(REFER),
            "benign_classes": list(BENIGN),
            "fit_split": "isic_train+non_isic_train(ddi,ddi2,fitzpatrick17k)",
            "test_melanoma_routed": isic_result["mel_routed"],
            "test_benign_referred": isic_result["benign_referred"],
            "test_auc": isic_result["auc"],
            "non_isic_test_auc": non_result["auc"],
            "non_isic_test_melanoma_routed": non_result["mel_routed"],
        }, indent=2))
        print(f"head written to {HEAD_OUT}")


if __name__ == "__main__":
    main()
