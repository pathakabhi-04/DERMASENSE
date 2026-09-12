"""
Step 1 of the phone-gap plan: does the CV-4b referral head's routing
survive the move from ISIC dermoscopy to real clinical photography?

Every strong number for the head (0.9024 melanoma routing @ 39.4%
benign referral) is measured on ISIC2019 test -- dermoscopic images,
taken with a dedicated instrument under controlled lighting. DDI and
DDI-2 (Stanford AIMI) are biopsy-proven CLINICAL photos: ordinary
cameras, real lighting, real skin tones -- the closest public stand-in
for phone-domain deployment. This script answers the one question that
matters before any retraining is considered: does the head still work
when the pixels no longer look like ISIC.

No training here. Same measurement primitive as the ISIC result: extract
2048-d backbone features through the frozen CV-4 classifier, score them
with the already-fitted ReferralHead, compare predicted referral against
ground truth. ~minutes, no GPU.

Pre-committed decision rule (set before this ran):
    melanoma routing >= 0.85 on the DDI+DDI2 combined melanoma set
        -> the head transfers; proceed to prospective phone validation
    0.70 <= routing < 0.85
        -> domain adaptation is justified, now with a metric to target
    routing < 0.70
        -> not phone-deployable yet; report this plainly

Known limitation, stated up front: DDI has 21 melanomas, DDI-2 has 4 --
25 combined. That is short of the ~50 that would give a tight interval;
the number below is directional, not a validated deployment metric.
Reported with a binomial confidence interval so the uncertainty is
visible rather than implied away.

## What "melanoma routing dropped" can actually mean -- and why this
## script checks which one it is

A dropped routing rate has two very different causes, and confusing
them leads to the wrong fix:

  1. The SCORE stops discriminating malignant from benign on this
     domain (the representation itself degrades) -- would need
     retraining / domain adaptation.
  2. The score still discriminates, but the THRESHOLD (fit on ISIC's
     score distribution) does not transfer, because clinical-photo
     scores sit on a different part of the range than dermoscopy scores
     -- would need only a recalibrated threshold, no retraining.

This script reports AUC (rank quality, threshold-independent) alongside
the operating-point pass rate specifically to distinguish these, plus
the overall referral rate: if that is close to the malignant-routed
rate, the head isn't discriminating at the shipped threshold at all --
it is referring almost everyone, which is the (2) failure mode.
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

from src.models.native_classifier import (
    DermaSenseNativeClassifier,
    NativeClassifierConfig,
)
from src.risk.referral_head import ReferralHead

REPO_ROOT = Path(__file__).resolve().parents[1]
DDI_DIR = REPO_ROOT / "data/raw/ddi"
DDI2_DIR = REPO_ROOT / "data/raw/ddi2"
CLASSIFIER_CHECKPOINT = (
    REPO_ROOT / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
)
HEAD_PATH = REPO_ROOT / "checkpoints/referral_head/referral_head.json"

# Same 224x224 ImageNet-normalized transform CV-4 uses for a crop
# (src/inference/orchestrator.py::_cv4_tensor, and matches
# src/data/transforms.py::build_eval_transform / _base_preprocessing
# exactly). These images are already lesion-centric clinical photos --
# the same convention PAD-UFES uses -- so no separate crop/detection
# step applies.
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval -- sane at small n, unlike a normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((centre - spread) / denom, (centre + spread) / denom)


def load_backbone(device: str = "cpu") -> torch.nn.Module:
    """
    Mirrors NativePredictor.from_checkpoint exactly (src/inference/native.py)
    rather than reconstructing the model independently: same backbone config
    (resnet50, pretrained=False -- the checkpoint supplies all weights) and
    strict=True, so a config mismatch raises instead of silently loading an
    architecturally-different model whose "features" would be garbage.
    """
    model = DermaSenseNativeClassifier(
        NativeClassifierConfig(backbone="resnet50", pretrained=False, dropout=0.0)
    )
    checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise RuntimeError("Could not find model state dict in checkpoint.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def extract_features(model: torch.nn.Module, image_path: Path, device: str) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = _TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.extract_features(tensor)
    return features.cpu().numpy().reshape(-1)


def load_ddi_labels() -> pd.DataFrame:
    df = pd.read_csv(DDI_DIR / "manifest.csv")
    df["image_path"] = df["DDI_file"].apply(lambda f: DDI_DIR / "images" / f)
    df["is_malignant"] = df["malignant"].astype(bool)
    df["is_melanoma"] = df["disease"].str.contains("melanoma", case=False, na=False)
    df["source"] = "DDI"
    return df[["image_path", "is_malignant", "is_melanoma", "source", "disease"]]


def load_ddi2_labels() -> pd.DataFrame:
    df = pd.read_csv(DDI2_DIR / "manifest.csv")
    # photo_id in the spreadsheet maps to a file in the image directory;
    # match by stem since extensions in the file table were .jpg.
    images_by_stem = {p.stem: p for p in (DDI2_DIR / "images").glob("*")}
    df["image_path"] = df["photo_id"].astype(str).apply(lambda s: images_by_stem.get(s))
    df = df[df["image_path"].notna()]
    df["is_malignant"] = df["benign_malignant"].str.lower() == "malignant"
    df["is_melanoma"] = df["diagnosis_detailed"].str.contains("melanoma", case=False, na=False)
    df["source"] = "DDI-2"
    return df[["image_path", "is_malignant", "is_melanoma", "source", "diagnosis_detailed"]].rename(
        columns={"diagnosis_detailed": "disease"}
    )


def main() -> None:
    device = "cpu"
    print("Loading CV-4 backbone and referral head...")
    model = load_backbone(device)
    head = ReferralHead.load(HEAD_PATH)

    frames = []
    if (DDI_DIR / "manifest.csv").exists():
        frames.append(load_ddi_labels())
    if (DDI2_DIR / "manifest.csv").exists():
        frames.append(load_ddi2_labels())

    if not frames:
        raise SystemExit("No DDI/DDI2 manifests found. Run scripts.acquire_ddi first.")

    data = pd.concat(frames, ignore_index=True)
    print(f"  {len(data)} labelled rows across {data['source'].unique().tolist()}")

    referred, probability, missing = [], [], 0
    for _, row in data.iterrows():
        path = row["image_path"]
        if path is None or not Path(path).exists():
            referred.append(None)
            probability.append(None)
            missing += 1
            continue
        features = extract_features(model, Path(path), device)
        if features is None:
            referred.append(None)
            probability.append(None)
            missing += 1
            continue
        decision = head.decide(features)
        referred.append(decision.refer)
        probability.append(decision.probability)

    data["referred"] = referred
    data["probability"] = probability
    if missing:
        print(f"  {missing} images unreadable/missing, excluded")
    data = data[data["referred"].notna()].copy()
    data["referred"] = data["referred"].astype(bool)
    data["probability"] = data["probability"].astype(float)

    print("\n" + "=" * 68)
    print("REFERRAL HEAD ON DDI + DDI-2 (clinical photos, not dermoscopy)")
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
    print(f"  COMBINED melanoma routing: {rate:.4f}  ({mel_routed}/{n_mel})  "
          f"95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"  COMBINED benign referred : {combined_ben['referred'].mean():.4f}  "
          f"({combined_ben['referred'].sum()}/{len(combined_ben)})")
    print(f"  (for reference, on ISIC2019 test: 0.9024 melanoma / 0.3940 benign)")

    overall_referred = data["referred"].mean()
    print(f"\n  OVERALL referral rate (any label): {overall_referred:.4f} "
          f"({data['referred'].sum()}/{len(data)})")
    print("  -- if this is close to the malignant-routed rate, the head is not")
    print("     discriminating at the shipped threshold on this domain, it is")
    print("     just referring almost everyone; that changes the diagnosis from")
    print("     'weaker signal' to 'operating point does not transfer'.")

    try:
        auc = roc_auc_score(data["is_malignant"].astype(int), data["probability"])
        print(f"\n  malignant-vs-benign AUC on DDI+DDI2 (domain-shifted probs): "
              f"{auc:.4f}  (ISIC test AUC was 0.9050)")
    except Exception as error:  # noqa: BLE001
        auc = None
        print(f"\n  could not compute AUC: {error}")

    if auc is not None and auc >= 0.75:
        print("  AUC is largely intact -> the SCORE still discriminates on this")
        print("  domain. The failure is the OPERATING POINT, not the head itself.")

        benign_scores = data.loc[~data["is_malignant"], "probability"].values
        redomain_threshold = float(np.quantile(benign_scores, 1 - 0.394))
        redomain_referred = data["probability"] >= redomain_threshold
        mel_mask = data["is_melanoma"]
        ben_mask = ~data["is_malignant"]
        print(f"\n  Re-picking the threshold ON THIS DOMAIN at the same 39.4% "
              f"benign-referral budget:")
        print(f"    threshold: {redomain_threshold:.4f} (was {head.threshold:.6f} from ISIC)")
        print(f"    melanoma routed : {redomain_referred[mel_mask].mean():.4f} "
              f"({redomain_referred[mel_mask].sum()}/{mel_mask.sum()})")
        print(f"    benign referred : {redomain_referred[ben_mask].mean():.4f}")
        print("  This threshold is fit ON the eval set and is illustrative only")
        print("  (not a valid held-out estimate) -- it answers 'is recalibration")
        print("  the right kind of fix', not 'here is the number to ship'.")

    print("\n  Pre-committed decision rule (on the ISIC-fit operating point):")
    if n_mel < 10:
        print(f"    n={n_mel} is too small for the rule to mean anything alone -- "
              "report directionally.")
    elif rate >= 0.85:
        print("    >=0.85 -> the head TRANSFERS. Proceed to prospective phone validation.")
    elif rate >= 0.70:
        print("    0.70-0.85 -> see the AUC/threshold analysis above before concluding")
        print("    'domain adaptation' is the fix -- it may only be recalibration.")
    else:
        print("    <0.70 -> NOT phone/clinical-photo deployable yet at this threshold.")

    out = REPO_ROOT / "analysis/quality/mel_sensitivity/referral_head_on_ddi.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    data.drop(columns=["image_path"]).to_csv(out, index=False)
    print(f"\n  per-image results written to {out}")


if __name__ == "__main__":
    main()
