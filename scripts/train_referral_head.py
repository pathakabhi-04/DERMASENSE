"""
Train the binary referral head properly: fit on ISIC val, evaluate on
ISIC test with test never touched during fitting.

Supersedes the out-of-fold probe in `probe_binary_separability.py`,
which measured decodability on the test split itself. That was the right
question then ("is the signal present?"). This is the shippable version
("does a head fitted elsewhere generalise to untouched test?").

The head predicts ONE thing: does this lesion need a clinician?
  refer  = MEL, BCC, SCC, AK   (URGENT_EVALUATION or EVALUATE_SOON)
  benign = NV, BKL             (MONITOR)
DF/VASC have no 6-class analogue and are excluded rather than mapped --
the taxonomy question stays open.

The operating point is chosen on VAL, then applied unchanged to test.
Choosing it on test would be selecting the answer.

    PYTHONPATH=. python3 scripts/train_referral_head.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data.torch_dataset import CVDatasetTorch

TRAIN_FEATURES = Path("analysis/scc_bcc/isic2019_train_backbone_features.npz")
VAL_FEATURES = Path("analysis/scc_bcc/isic2019_val_backbone_features.npz")
TEST_FEATURES = Path("analysis/scc_bcc/isic2019_test_scc_bcc_features.npz")
OUT = Path("analysis/quality/mel_sensitivity/referral_head_result.json")

REFER = ("MEL", "BCC", "SCC", "AK")
BENIGN = ("NV", "BKL")
# Chosen on val: the benign-referral budget we are willing to spend.
TARGET_BENIGN_REFERRAL = 0.30


def load(split: str, path: Path):
    data = np.load(path)
    ds = CVDatasetTorch(dataset_id="isic2019", split=split, verify_images=False)
    diagnosis = np.array([ds.get_diagnosis(i) for i in range(len(ds))])
    assert len(diagnosis) == len(data["features"]), f"{split} length mismatch"
    keep = np.isin(diagnosis, REFER + BENIGN)
    return (
        data["features"][keep],
        np.isin(diagnosis[keep], REFER).astype(int),
        diagnosis[keep],
    )


def main() -> None:
    # Fit on TRAIN (18,402). Val is used only to choose the operating
    # point, test only to report. 3,304 val samples in 2,048 dimensions
    # was a thin regime -- this is the 5.5x version of the same fit.
    Xtr, ytr, _ = load("train", TRAIN_FEATURES)
    Xv, yv, dv = load("val", VAL_FEATURES)
    Xt, yt, dt = load("test", TEST_FEATURES)
    print(f"  fit on train: {len(ytr)} ({ytr.sum()} refer / {(ytr == 0).sum()} benign)")
    print(f"  threshold on val: {len(yv)}")
    print(f"  eval on test: {len(yt)} ({yt.sum()} refer / {(yt == 0).sum()} benign)")
    print(f"  test melanomas: {(dt == 'MEL').sum()}\n")

    head = LogisticRegression(max_iter=3000, C=1.0)
    head.fit(Xtr, ytr)

    # Operating point fixed on VAL, then applied to test unchanged.
    val_scores = head.predict_proba(Xv)[:, 1]
    threshold = float(np.quantile(val_scores[yv == 0], 1 - TARGET_BENIGN_REFERRAL))
    print(f"  operating point chosen on val: {threshold:.4f} "
          f"(= {TARGET_BENIGN_REFERRAL:.0%} benign referral on val)\n")

    test_scores = head.predict_proba(Xt)[:, 1]
    referred = test_scores >= threshold

    mel = dt == "MEL"
    mel_routed = float(referred[mel].mean())
    benign_referred = float(referred[yt == 0].mean())
    auc = float(roc_auc_score(yt, test_scores))

    print("=" * 68)
    print("HELD-OUT TEST (never used for fitting or threshold selection)")
    print("=" * 68)
    print(f"  melanoma routed to a clinician : {mel_routed:.4f}  "
          f"({int(referred[mel].sum())}/{int(mel.sum())})")
    print(f"  benign referred                : {benign_referred:.4f}")
    print(f"  refer-vs-benign AUC            : {auc:.4f}")

    print("\n  versus the shipped 6-class head on the same test set:")
    print(f"    {'argmax (shipped)':<28} 0.7180 melanoma @ 0.2320 benign")
    print(f"    {'best prob. threshold':<28} 0.8060 melanoma @ 0.3800 benign")
    print(f"    {'referral head (this)':<28} {mel_routed:.4f} melanoma @ {benign_referred:.4f} benign")

    cleared = mel_routed >= 0.90
    print(f"\n  >=0.90 melanoma routing target: {'MET' if cleared else 'NOT met'}")
    print("=" * 68)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "fit_split": "train", "threshold_split": "val", "eval_split": "test",
        "threshold_chosen_on": "val", "threshold": threshold,
        "test_melanoma_routed": mel_routed,
        "test_benign_referred": benign_referred,
        "test_auc": auc, "target_met": bool(cleared),
    }, indent=2))
    print(f"  written to {OUT}")


if __name__ == "__main__":
    main()
