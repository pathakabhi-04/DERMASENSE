"""
Scoping check before any retraining/adaptation decision: does the
domain-transfer collapse found for CV-4b (referral_head_on_ddi.md,
AUC 0.9050 -> 0.5742) also affect the ACTUAL PRODUCT PATH -- the native
6-way classifier's own softmax and the shipped diagnosis_to_action
routing -- or is it specific to the referral head as a separate linear
probe fit only on ISIC features?

This matters because the two heads have different training histories on
the SAME backbone checkpoint:
  - The 6-way native head was fine-tuned end-to-end ON PAD-UFES, which
    IS real smartphone/clinical photography (docs/CV_DATASET_SPEC_v1.0.md).
  - The referral head is a linear probe fit AFTER that, using only
    ISIC-domain features -- it never saw a PAD-UFES-domain feature
    during its own fitting, only ISIC ones (scripts/train_referral_head.py).

So the collapse could be (a) referral-head-specific overfitting to an
ISIC-only feature distribution, cheaply fixable by refitting the probe
on broader data -- or (b) the shared backbone representation itself not
carrying malignant/benign structure on clinical photography at all, in
which case retraining just the referral head solves nothing and the
real problem is upstream (docs/cv_metrics_improvement_plan.md Step 2).

No training here either. Reuses NativePredictor -- the exact class
`DermaSensePipeline` calls in production (src/inference/orchestrator.py)
-- rather than any reimplementation, so this measures the real product
path, not an approximation of it.

Directly comparable numbers already on record:
  ISIC2019 test, argmax (shipped 6-class): 0.7180 melanoma routed
    (analysis/quality/mel_sensitivity/pad_vs_isic_routing.md)
  ISIC2019 test, referral head:            0.9024 melanoma routed,
                                            AUC 0.9050
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from src.inference.native import NativePredictor
from src.risk.action_mapping import HIGH_RISK_DIAGNOSES

REPO_ROOT = Path(__file__).resolve().parents[1]
DDI_DIR = REPO_ROOT / "data/raw/ddi"
DDI2_DIR = REPO_ROOT / "data/raw/ddi2"
CLASSIFIER_CHECKPOINT = (
    REPO_ROOT / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
)

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((centre - spread) / denom, (centre + spread) / denom)


def load_ddi_labels() -> pd.DataFrame:
    df = pd.read_csv(DDI_DIR / "manifest.csv")
    df["image_path"] = df["DDI_file"].apply(lambda f: DDI_DIR / "images" / f)
    df["is_malignant"] = df["malignant"].astype(bool)
    df["is_melanoma"] = df["disease"].str.contains("melanoma", case=False, na=False)
    df["source"] = "DDI"
    return df[["image_path", "is_malignant", "is_melanoma", "source"]]


def load_ddi2_labels() -> pd.DataFrame:
    df = pd.read_csv(DDI2_DIR / "manifest.csv")
    images_by_stem = {p.stem: p for p in (DDI2_DIR / "images").glob("*")}
    df["image_path"] = df["photo_id"].astype(str).apply(lambda s: images_by_stem.get(s))
    df = df[df["image_path"].notna()]
    df["is_malignant"] = df["benign_malignant"].str.lower() == "malignant"
    df["is_melanoma"] = df["diagnosis_detailed"].str.contains("melanoma", case=False, na=False)
    df["source"] = "DDI-2"
    return df[["image_path", "is_malignant", "is_melanoma", "source"]]


def main() -> None:
    print("Loading the actual product classifier (NativePredictor.from_checkpoint)...")
    predictor = NativePredictor.from_checkpoint(CLASSIFIER_CHECKPOINT, device="cpu")

    frames = []
    if (DDI_DIR / "manifest.csv").exists():
        frames.append(load_ddi_labels())
    if (DDI2_DIR / "manifest.csv").exists():
        frames.append(load_ddi2_labels())
    data = pd.concat(frames, ignore_index=True)
    print(f"  {len(data)} labelled rows across {data['source'].unique().tolist()}")

    malignant_score, is_referred, predicted_class, missing = [], [], [], 0
    for _, row in data.iterrows():
        path = row["image_path"]
        image = cv2.imread(str(path)) if path is not None else None
        if image is None:
            malignant_score.append(None)
            is_referred.append(None)
            predicted_class.append(None)
            missing += 1
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = _TRANSFORM(Image.fromarray(rgb))
        prediction = predictor.predict(tensor)

        score = sum(prediction.probabilities[c] for c in HIGH_RISK_DIAGNOSES)
        malignant_score.append(score)
        # "referred" here means the shipped system does NOT tell the user
        # MONITOR/low-risk -- i.e. argmax lands outside MONITOR_DIAGNOSES.
        is_referred.append(prediction.predicted_class not in {"NEV", "SEK"})
        predicted_class.append(prediction.predicted_class)

    data["malignant_score"] = malignant_score
    data["referred"] = is_referred
    data["predicted_class"] = predicted_class
    if missing:
        print(f"  {missing} images unreadable/missing, excluded")
    data = data[data["referred"].notna()].copy()
    data["referred"] = data["referred"].astype(bool)
    data["malignant_score"] = data["malignant_score"].astype(float)

    print("\n" + "=" * 68)
    print("NATIVE 6-CLASS CLASSIFIER (shipped product path) ON DDI + DDI-2")
    print("=" * 68)

    for source in data["source"].unique():
        sub = data[data["source"] == source]
        mal = sub[sub["is_malignant"]]
        ben = sub[~sub["is_malignant"]]
        mel = sub[sub["is_melanoma"]]
        print(f"\n  {source}: n={len(sub)}  malignant={len(mal)}  benign={len(ben)}  melanoma={len(mel)}")
        if len(mal):
            print(f"    malignant routed : {mal['referred'].mean():.4f} ({mal['referred'].sum()}/{len(mal)})")
        if len(ben):
            print(f"    benign referred  : {ben['referred'].mean():.4f} ({ben['referred'].sum()}/{len(ben)})")
        if len(mel):
            lo, hi = _wilson_interval(int(mel["referred"].sum()), len(mel))
            print(f"    MELANOMA routed  : {mel['referred'].mean():.4f} "
                  f"({mel['referred'].sum()}/{len(mel)})  95% CI [{lo:.2f}, {hi:.2f}]")

    combined_mel = data[data["is_melanoma"]]
    combined_ben = data[~data["is_malignant"]]
    n_mel = len(combined_mel)
    mel_routed = combined_mel["referred"].sum() if n_mel else 0
    rate = mel_routed / n_mel if n_mel else float("nan")
    lo, hi = _wilson_interval(int(mel_routed), n_mel)

    print("\n" + "-" * 68)
    print(f"  COMBINED melanoma routing (argmax != MONITOR): {rate:.4f}  "
          f"({mel_routed}/{n_mel})  95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"  COMBINED benign referred                     : "
          f"{combined_ben['referred'].mean():.4f} ({combined_ben['referred'].sum()}/{len(combined_ben)})")

    overall = data["referred"].mean()
    print(f"  OVERALL referral rate (any label)             : {overall:.4f} "
          f"({data['referred'].sum()}/{len(data)})")

    auc = roc_auc_score(data["is_malignant"].astype(int), data["malignant_score"])
    print(f"\n  malignant-vs-benign AUC (native softmax mass): {auc:.4f}")

    print("\n  REFERENCE POINTS (same images, different scorer):")
    print(f"    referral head (CV-4b) on this same DDI+DDI2 set: AUC 0.5742, "
          f"melanoma routed 0.7600")
    print(f"    ISIC2019 test, argmax (this classifier, in-domain): "
          f"melanoma routed 0.7180")
    print(f"    ISIC2019 test, referral head (in-domain): AUC 0.9050, "
          f"melanoma routed 0.9024")

    print("\n  READ THIS AS:")
    print(f"    if native-softmax AUC ({auc:.4f}) is ALSO near chance -> the shared")
    print("    backbone itself doesn't transfer; referral-head-only retraining")
    print("    would not fix the underlying problem.")
    print(f"    if native-softmax AUC ({auc:.4f}) holds up notably better than the")
    print("    referral head's 0.5742 -> the collapse is specific to the referral")
    print("    head's ISIC-only fitting, and refitting/retraining THAT probe on")
    print("    broader data is the targeted, cheap fix.")

    out = REPO_ROOT / "analysis/quality/mel_sensitivity/native_classifier_on_ddi.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    data.drop(columns=["image_path"]).to_csv(out, index=False)
    print(f"\n  per-image results written to {out}")


if __name__ == "__main__":
    main()
