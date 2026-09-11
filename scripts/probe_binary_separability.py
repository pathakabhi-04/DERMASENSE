"""
Does the backbone already know what the 6-class head gets wrong?

78 of 666 ISIC-test melanomas receive <5% total malignant probability
from the deployed 6-class head -- confidently wrong, unrecoverable by any
threshold (analysis/quality/mel_sensitivity/threshold_vs_retraining.md).

Before spending GPU on a binary malignant/benign head, this asks whether
the information is even present in the representation: train a LINEAR
probe on the cached penultimate features and see where it puts those 78.

If a linear probe recovers them, a trained head will do at least as well
and retraining is justified. If it cannot, the signal is absent from the
representation and no head will invent it -- which redirects the effort
to data (domain-appropriate photos) rather than objective.

Costs no training and no GPU: features are already cached from the SCC/BCC
investigation, and a logistic probe on 3.5k x 2048 takes seconds.

    PYTHONPATH=. python3 scripts/probe_binary_separability.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.data.torch_dataset import CVDatasetTorch
from src.inference.native import NativePredictor

FEATURES = Path("analysis/scc_bcc/isic2019_test_scc_bcc_features.npz")
CHECKPOINT = Path("checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt")
OUT = Path("analysis/quality/mel_sensitivity/binary_probe_result.json")

MALIGNANT_6 = ("MEL", "BCC", "SCC")
# "Needs a clinician" in ISIC terms. AK is precancerous and routes to
# EVALUATE_SOON, so it counts as referral. DF/VASC have no 6-class
# analogue and are excluded rather than mapped.
REFER = ("MEL", "BCC", "SCC", "AK")
BENIGN = ("NV", "BKL")
HARD_THRESHOLD = 0.05


def main() -> None:
    data = np.load(FEATURES)
    features, targets = data["features"], data["targets"]

    dataset = CVDatasetTorch(dataset_id="isic2019", split="test", verify_images=False)
    names = list(dataset.class_names)
    diagnosis = np.array([dataset.get_diagnosis(i) for i in range(len(dataset))])

    assert len(diagnosis) == len(features), "feature/dataset length mismatch"

    # --- which melanomas does the 6-class head get confidently wrong? ---
    print("Scoring melanomas with the deployed 6-class head...")
    predictor = NativePredictor.from_checkpoint(CHECKPOINT, device="cpu")
    mel_idx = np.where(diagnosis == "MEL")[0]

    p_mal = np.zeros(len(mel_idx))
    for n, i in enumerate(mel_idx):
        with torch.no_grad():
            r = predictor.predict(dataset[int(i)]["image"])
        p_mal[n] = sum(r.probabilities.get(c, 0.0) for c in MALIGNANT_6)
        if (n + 1) % 250 == 0:
            print(f"  ...{n + 1}/{len(mel_idx)}")

    hard_mask = p_mal < HARD_THRESHOLD
    hard_idx = mel_idx[hard_mask]
    print(f"\n  melanomas with p_malignant < {HARD_THRESHOLD}: {len(hard_idx)}")

    # --- linear probe: refer vs benign, out-of-fold ---
    keep = np.isin(diagnosis, REFER + BENIGN)
    X, y = features[keep], np.isin(diagnosis[keep], REFER).astype(int)
    positions = np.where(keep)[0]

    print(f"  probe set: {keep.sum()} images  ({y.sum()} refer / {(~y.astype(bool)).sum()} benign)")

    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]

    pos_of = {p: k for k, p in enumerate(positions)}
    hard_scores = np.array([oof[pos_of[i]] for i in hard_idx if i in pos_of])
    benign_scores = oof[y == 0]
    all_mel_scores = np.array([oof[pos_of[i]] for i in mel_idx if i in pos_of])

    print("\n" + "=" * 70)
    print("LINEAR PROBE on cached backbone features (out-of-fold)")
    print("=" * 70)
    print(f"  {'operating point':>18}  {'all MEL':>9}  {'the 78':>9}  {'benign referred':>16}")
    print("  " + "-" * 62)
    rows = []
    for q in (0.10, 0.20, 0.30, 0.40, 0.50):
        thr = float(np.quantile(benign_scores, 1 - q))
        mel_rate = float((all_mel_scores >= thr).mean())
        hard_rate = float((hard_scores >= thr).mean())
        rows.append({"benign_referred": q, "threshold": thr,
                     "all_mel_recovered": mel_rate, "hard_recovered": hard_rate})
        flag = "  <-- meets rule" if hard_rate >= 0.50 and q <= 0.40 else ""
        print(f"  {q:>17.0%}  {mel_rate:>9.3f}  {hard_rate:>9.3f}  {q:>16.0%}{flag}")

    best = max((r for r in rows if r["benign_referred"] <= 0.40),
               key=lambda r: r["hard_recovered"])
    verdict = best["hard_recovered"] >= 0.50

    print("\n" + "=" * 70)
    print(f"  Best at <=40% benign referral: {best['hard_recovered']:.1%} of the 78 recovered")
    print(f"  VERDICT: {'signal IS present -- a binary head is justified' if verdict else 'signal is ABSENT -- a new head will not recover them'}")
    print("=" * 70)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_hard": int(len(hard_idx)), "sweep": rows,
        "best_at_40pct": best, "signal_present": bool(verdict)}, indent=2))
    print(f"  written to {OUT}")


if __name__ == "__main__":
    main()
