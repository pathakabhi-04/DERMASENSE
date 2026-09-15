"""
Do mask-derived geometric features (the ABCD rule) carry melanoma signal
that SURVIVES domain shift, where the CNN's learned features did not?

## Why this is worth one experiment

Four investigations showed the backbone's features collapse on unseen
capture sources (0.63-0.67 held-out). The stated mechanism was lighting,
white balance and compression -- all of which change *appearance*.
Shape does not. A lesion's asymmetry and border irregularity are
properties of its outline, and CV-3 produces that outline reliably
(0.8-2.6% degenerate masks across four unseen sources).

So the hypothesis is specific: **geometry may transfer where texture did
not.**

## Scope corrections made before running, not after

- **ABCDE is a melanoma rule**, not a general malignancy rule. BCC and
  SCC do not follow it. The primary comparison is therefore
  **MEL vs NEV+SEK**, not malignant-vs-benign. The malignant-vs-benign
  number is reported too, but only because it is what the CNN baselines
  were measured on.
- **D (diameter) is not computable.** It needs physical scale, which
  exists only when a ruler is in frame. This measures **A, B, C**.
- "E (evolving)" is CV-7 and needs a prior image by definition.

## Features, all from the mask and the pixels under it

  A  asymmetry          fold the mask about its two principal axes;
                        non-overlapping fraction, averaged
  B  border irregularity 1 - solidity (area / convex-hull area), plus
                        1 - circularity (4*pi*area / perimeter^2)
  C  colour variation    spread of LAB values inside the mask

## Pre-committed design

Leave-one-source-group-out, three groups: `stanford` (DDI+DDI-2),
`atlas` (Fitzpatrick17k), `pad` (PAD-UFES). Fit a logistic model on two,
test on the third, so the transfer question is asked directly rather
than inferred.

  Primary metric  MEL-vs-(NEV+SEK) AUC on the HELD-OUT source group
  Bar             >= 0.70 on ALL THREE held-out groups, AND >= the
                  referral head's AUC on the same data
  Baseline        CNN referral head, malignant-vs-benign, measured on
                  these same sources: DDI 0.5900, DDI-2 0.6120,
                  Fitzpatrick 0.6924 (domain_shift_second_source.md)

  If it clears the bar -> geometry transfers where texture did not, and
  Plan A has a defensible first-visit feature (stated one-directionally:
  flag concerning features, NEVER affirm reassuring ones).
  If it does not -> the first-visit gap stands, and this closes.

**Even if it succeeds it is not a risk score.** A model that predicts
melanoma accurately is a classifier wearing different clothes, and Plan
A does not ship those. The honest use is "these features are worth
mentioning to your doctor".

    PYTHONPATH=. python3 scripts/evaluate_abcd_features.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.inference.crop_normalize import crop_and_normalize
from src.segmentation.inference import load_segmentation_model, mask_evidence, predict_mask

REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_CHECKPOINT = REPO_ROOT / "checkpoints/cv3_512/best.pt"
BUNDLE = REPO_ROOT / "data/processed/cv4b_bundle"
EXTERNAL_SET = REPO_ROOT / "analysis/quality/mel_sensitivity/external_6class_eval_set.csv"
OUT = REPO_ROOT / "analysis/quality/mel_sensitivity/abcd_features.csv"

FULL_FRAME_BOX = (0.5, 0.5, 1.0, 1.0)
BENIGN_CLASSES = ("NEV", "SEK")
SOURCE_GROUP = {"ddi": "stanford", "ddi2": "stanford",
                "fitzpatrick17k": "atlas", "pad_ufes": "pad"}
# domain_shift_second_source.md, malignant-vs-benign, same images
REFERRAL_HEAD_BASELINE = {"stanford": 0.60, "atlas": 0.6924, "pad": float("nan")}
FEATURES = ("asymmetry", "border_solidity", "border_circularity", "colour_variation")
BAR = 0.70


def abc_features(image_bgr: np.ndarray, mask: np.ndarray) -> dict | None:
    """A, B and C from one lesion outline. Returns None on a mask CV-3
    could not resolve -- those are excluded and counted, not imputed."""
    binary = (mask > 0.5).astype(np.uint8)
    if binary.sum() < 50:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0 or perimeter <= 0:
        return None

    # --- B: border irregularity, two independent formulations ---
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0 else 1.0
    circularity = 4 * np.pi * area / (perimeter ** 2)

    # --- A: asymmetry about the principal axes ---
    moments = cv2.moments(binary, binaryImage=True)
    if moments["m00"] == 0:
        return None
    cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]
    angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

    height, width = binary.shape
    rotation = cv2.getRotationMatrix2D((cx, cy), np.degrees(angle), 1.0)
    aligned = cv2.warpAffine(binary, rotation, (width, height), flags=cv2.INTER_NEAREST)
    total = aligned.sum()
    if total == 0:
        return None
    asymmetries = []
    for flip_axis in (1, 0):  # about the vertical, then the horizontal axis
        mismatch = np.logical_xor(aligned, np.flip(aligned, axis=flip_axis)).sum()
        asymmetries.append(mismatch / (2.0 * total))

    # --- C: colour variation inside the lesion ---
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    inside = lab[binary.astype(bool)]
    colour_variation = float(inside.std(axis=0).mean()) if len(inside) else 0.0

    return {
        "asymmetry": float(np.mean(asymmetries)),
        "border_solidity": float(1.0 - solidity),
        "border_circularity": float(1.0 - circularity),
        "colour_variation": colour_variation,
    }


def build_table() -> pd.DataFrame:
    rows = []
    external = pd.read_csv(EXTERNAL_SET)
    for row in external.itertuples():
        stem = Path(row.image_path).stem
        path = BUNDLE / "images" / row.source / f"{stem}.jpg"
        if path.exists():
            rows.append({"path": str(path), "label": row.label, "source": row.source})

    # PAD-UFES as a third source group -- it carries the same 6-class
    # labels natively, so no taxonomy mapping is involved.
    for split in ("train", "val", "test"):
        pad = pd.read_csv(REPO_ROOT / f"data/splits/pad_ufes/{split}.csv")
        for row in pad.itertuples():
            label = str(row.native_diagnosis).upper()
            path = BUNDLE / "images/pad_ufes" / f"{Path(row.image_path).stem}.jpg"
            if path.exists():
                rows.append({"path": str(path), "label": label, "source": "pad_ufes"})

    table = pd.DataFrame(rows)
    table["domain_group"] = table["source"].map(SOURCE_GROUP)
    table["is_melanoma"] = table["label"] == "MEL"
    table["is_benign"] = table["label"].isin(BENIGN_CLASSES)
    table["is_malignant"] = ~table["is_benign"]
    return table


def extract(table: pd.DataFrame) -> pd.DataFrame:
    device = torch.device("cpu")
    model = load_segmentation_model(SEGMENTATION_CHECKPOINT, device)
    records, skipped = [], 0
    for position, row in enumerate(table.itertuples(), start=1):
        image = cv2.imread(row.path, cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            continue
        tensor, _ = crop_and_normalize(image, FULL_FRAME_BOX, margin=0.0)
        mask = predict_mask(model, tensor, device)
        evidence = mask_evidence(mask)
        features = abc_features(image, mask)
        if features is None or evidence["mask_degenerate"]:
            skipped += 1
            continue
        records.append({
            "path": row.path, "label": row.label, "source": row.source,
            "domain_group": row.domain_group, "is_melanoma": row.is_melanoma,
            "is_benign": row.is_benign, "is_malignant": row.is_malignant,
            **features,
        })
        if position % 250 == 0 or position == len(table):
            print(f"  {position}/{len(table)}  (skipped {skipped})", flush=True)
    print(f"  extracted {len(records)}, skipped {skipped} (unreadable or degenerate mask)")
    return pd.DataFrame(records)


def loso_auc(data: pd.DataFrame, positive: str, negative_mask) -> dict:
    """Fit on two source groups, score the third. Same discipline as the
    CNN experiments, so the numbers are comparable."""
    results = {}
    for held_out in sorted(data["domain_group"].dropna().unique()):
        train = data[data["domain_group"] != held_out]
        test = data[data["domain_group"] == held_out]
        train = train[train[positive] | negative_mask(train)]
        test = test[test[positive] | negative_mask(test)]
        # An AUC computed against a handful of negatives is noise, not a
        # measurement. Fitzpatrick17k's mapped set is 83 melanomas and ONE
        # benign lesion, and scored 0.8675 before this guard existed --
        # the single number that appeared to clear the bar.
        MIN_PER_CLASS = 10
        positives, negatives = int(test[positive].sum()), int((~test[positive]).sum())
        if (test[positive].nunique() < 2 or train[positive].nunique() < 2
                or min(positives, negatives) < MIN_PER_CLASS):
            results[held_out] = float("nan")
            results[f"{held_out}_n"] = len(test)
            results[f"{held_out}_pos"] = positives
            results[f"{held_out}_neg"] = negatives
            continue
        scaler = StandardScaler().fit(train[list(FEATURES)])
        model = LogisticRegression(max_iter=2000).fit(
            scaler.transform(train[list(FEATURES)]), train[positive].astype(int)
        )
        scores = model.predict_proba(scaler.transform(test[list(FEATURES)]))[:, 1]
        results[held_out] = float(roc_auc_score(test[positive].astype(int), scores))
        results[f"{held_out}_n"] = len(test)
        results[f"{held_out}_pos"] = positives
        results[f"{held_out}_neg"] = negatives
    return results


def main() -> None:
    table = build_table()
    print(f"{len(table)} images: {table['domain_group'].value_counts().to_dict()}")
    print(f"  melanoma {int(table['is_melanoma'].sum())}, "
          f"benign(NEV+SEK) {int(table['is_benign'].sum())}\n")

    data = extract(table)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT, index=False)
    print(f"\nfeatures -> {OUT}\n")

    print("=" * 76)
    print("PRIMARY: melanoma vs benign (NEV+SEK) -- the actual ABCDE question")
    print("=" * 76)
    print("\nsingle features, pooled (direction check):")
    pooled = data[data["is_melanoma"] | data["is_benign"]]
    for feature in FEATURES:
        auc = roc_auc_score(pooled["is_melanoma"].astype(int), pooled[feature])
        print(f"  {feature:22} AUC {auc:.4f}")

    print("\nleave-one-source-out, 4-feature model:")
    primary = loso_auc(data, "is_melanoma", lambda d: d["is_benign"])
    for group in ("stanford", "atlas", "pad"):
        auc = primary.get(group, float("nan"))
        baseline = REFERRAL_HEAD_BASELINE.get(group, float("nan"))
        flag = ("  UNMEASURABLE (too few of one class)" if np.isnan(auc)
                else ("  >= bar" if auc >= BAR else "  BELOW bar"))
        print(f"  held out {group:10} AUC {auc:.4f}  "
              f"(mel={primary.get(f'{group}_pos', 0)}, "
              f"benign={primary.get(f'{group}_neg', 0)})"
              f"  CNN head {baseline:.4f}{flag}")

    print("\n" + "=" * 76)
    print("SECONDARY: malignant vs benign -- only for comparability with the CNN")
    print("=" * 76)
    secondary = loso_auc(data, "is_malignant", lambda d: d["is_benign"])
    for group in ("stanford", "atlas", "pad"):
        auc = secondary.get(group, float("nan"))
        print(f"  held out {group:10} AUC {auc:.4f}  "
              f"CNN head {REFERRAL_HEAD_BASELINE.get(group, float('nan')):.4f}")

    cleared = [primary.get(g, float("nan")) for g in ("stanford", "atlas", "pad")]
    passed = all(not np.isnan(a) and a >= BAR for a in cleared)
    print("\n" + "=" * 76)
    print(f"DECISION: bar was >= {BAR} on ALL THREE held-out groups")
    print("VERDICT:", "CLEARS -- geometry transfers where texture did not"
          if passed else "DOES NOT CLEAR -- the first-visit gap stands")
    print("=" * 76)


if __name__ == "__main__":
    main()
