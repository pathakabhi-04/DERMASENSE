"""
Shared domain-shift evaluation: run both the native 6-class classifier
(the shipped product path) and the CV-4b referral head against a labelled
image set, and report the same comparison for any source.

Written after doing this twice by hand for DDI/DDI-2
(referral_head_on_ddi.md, native_classifier_on_ddi.md) -- this
consolidates that logic so a THIRD source (or a fourth, later) is one
function away rather than a near-duplicate script. See those two docs
for the full reasoning on why both scorers are reported (AUC vs
operating-point pass rate answer different questions -- see
native_classifier_on_ddi.md's "read this as" section) and why AUC
specifically, not just the routing rate, is what distinguishes a
representation problem from a threshold problem.

Sources implement a small adapter protocol: a function that returns a
DataFrame with columns [image_path, is_malignant, is_melanoma, source].
Add one for a new dataset rather than writing a new evaluation script.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from src.inference.native import NativePredictor
from src.models.native_classifier import (
    DermaSenseNativeClassifier,
    NativeClassifierConfig,
)
from src.risk.action_mapping import HIGH_RISK_DIAGNOSES
from src.risk.referral_head import ReferralHead
from src.training.metrics import wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_CHECKPOINT = (
    REPO_ROOT / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
)
HEAD_PATH = REPO_ROOT / "checkpoints/referral_head/referral_head.json"

# Matches src/data/transforms.py::_base_preprocessing exactly (verified
# during the DDI evaluation): Resize((224,224)), ToTensor, ImageNet norm.
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_backbone_for_features(device: str = "cpu") -> torch.nn.Module:
    """For the referral head, which needs raw 2048-d features, not the
    classifier's own softmax. Mirrors NativePredictor.from_checkpoint's
    construction exactly (backbone config + strict=True) -- see
    referral_head_on_ddi.md for why this was checked rather than assumed."""
    model = DermaSenseNativeClassifier(
        NativeClassifierConfig(backbone="resnet50", pretrained=False, dropout=0.0)
    )
    checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def _read_tensor(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return _TRANSFORM(Image.fromarray(rgb))


def evaluate(data: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    `data` needs columns: image_path, is_malignant, is_melanoma.
    Returns `data` with `native_referred`, `native_malignant_score`,
    `head_referred`, `head_probability` columns added; unreadable rows
    are dropped and reported.
    """
    device = "cpu"
    print(f"[{source_label}] loading models...")
    predictor = NativePredictor.from_checkpoint(CLASSIFIER_CHECKPOINT, device=device)
    backbone = load_backbone_for_features(device)
    head = ReferralHead.load(HEAD_PATH)

    native_referred, native_score = [], []
    head_referred, head_prob = [], []
    missing = 0

    for _, row in data.iterrows():
        tensor = _read_tensor(Path(row["image_path"]))
        if tensor is None:
            native_referred.append(None); native_score.append(None)
            head_referred.append(None); head_prob.append(None)
            missing += 1
            continue

        prediction = predictor.predict(tensor)
        native_referred.append(prediction.predicted_class not in {"NEV", "SEK"})
        native_score.append(sum(prediction.probabilities[c] for c in HIGH_RISK_DIAGNOSES))

        with torch.no_grad():
            features = backbone.extract_features(tensor.unsqueeze(0).to(device))
        decision = head.decide(features.cpu().numpy().reshape(-1))
        head_referred.append(decision.refer)
        head_prob.append(decision.probability)

    data = data.copy()
    data["native_referred"] = native_referred
    data["native_malignant_score"] = native_score
    data["head_referred"] = head_referred
    data["head_probability"] = head_prob
    if missing:
        print(f"  {missing} images unreadable/missing, excluded")

    data = data[data["native_referred"].notna()].copy()
    for col in ("native_referred", "head_referred"):
        data[col] = data[col].astype(bool)
    for col in ("native_malignant_score", "head_probability"):
        data[col] = data[col].astype(float)
    return data


def report(data: pd.DataFrame, source_label: str) -> dict:
    mel = data[data["is_melanoma"]]
    ben = data[~data["is_malignant"]]
    mal = data[data["is_malignant"]]

    print(f"\n{'='*68}\n{source_label}: n={len(data)}  malignant={len(mal)}  "
          f"benign={len(ben)}  melanoma={len(mel)}\n{'='*68}")

    results = {"source": source_label, "n": len(data), "n_melanoma": len(mel)}

    for scorer, referred_col, score_col in (
        ("native classifier", "native_referred", "native_malignant_score"),
        ("referral head", "head_referred", "head_probability"),
    ):
        overall = data[referred_col].mean()
        mel_rate = mel[referred_col].mean() if len(mel) else float("nan")
        ben_rate = ben[referred_col].mean() if len(ben) else float("nan")
        auc = roc_auc_score(data["is_malignant"].astype(int), data[score_col]) \
            if data["is_malignant"].nunique() > 1 else float("nan")
        lo, hi = wilson_interval(int(mel[referred_col].sum()), len(mel)) if len(mel) else (0, 0)

        print(f"\n  {scorer}:")
        print(f"    overall referred : {overall:.4f}")
        print(f"    benign referred  : {ben_rate:.4f}")
        print(f"    melanoma routed  : {mel_rate:.4f} ({mel[referred_col].sum()}/{len(mel)})  "
              f"95% CI [{lo:.2f}, {hi:.2f}]")
        print(f"    malignant-vs-benign AUC: {auc:.4f}")

        key = scorer.replace(" ", "_")
        results.update({
            f"{key}_overall_referred": overall, f"{key}_benign_referred": ben_rate,
            f"{key}_melanoma_routed": mel_rate, f"{key}_melanoma_ci_lo": lo,
            f"{key}_melanoma_ci_hi": hi, f"{key}_auc": auc,
        })

    return results
