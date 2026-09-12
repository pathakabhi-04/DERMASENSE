"""
Run the domain-shift check across DDI, DDI-2, and Fitzpatrick17k
(atlasdermatologico.com.br subset), and report side by side.

This is the second-source check called for in
native_classifier_on_ddi.md: is the collapse found on DDI/DDI-2
(referral head AUC 0.90->0.57, native classifier melanoma routing
0.72->0.32) a property of that one Stanford-sourced dataset, or does it
reproduce on an independently-curated clinical-photo source?

    python -m scripts.run_domain_shift_check
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.evaluate_domain_shift import evaluate, report

REPO_ROOT = Path(__file__).resolve().parents[1]
DDI_DIR = REPO_ROOT / "data/raw/ddi"
DDI2_DIR = REPO_ROOT / "data/raw/ddi2"
FITZ_DIR = REPO_ROOT / "data/raw/fitzpatrick17k"


def load_ddi() -> pd.DataFrame:
    df = pd.read_csv(DDI_DIR / "manifest.csv")
    df["image_path"] = df["DDI_file"].apply(lambda f: DDI_DIR / "images" / f)
    df["is_malignant"] = df["malignant"].astype(bool)
    df["is_melanoma"] = df["disease"].str.contains("melanoma", case=False, na=False)
    return df[["image_path", "is_malignant", "is_melanoma"]]


def load_ddi2() -> pd.DataFrame:
    df = pd.read_csv(DDI2_DIR / "manifest.csv")
    images_by_stem = {p.stem: p for p in (DDI2_DIR / "images").glob("*")}
    df["image_path"] = df["photo_id"].astype(str).apply(lambda s: images_by_stem.get(s))
    df = df[df["image_path"].notna()]
    df["is_malignant"] = df["benign_malignant"].str.lower() == "malignant"
    df["is_melanoma"] = df["diagnosis_detailed"].str.contains("melanoma", case=False, na=False)
    return df[["image_path", "is_malignant", "is_melanoma"]]


def load_fitzpatrick() -> pd.DataFrame:
    df = pd.read_csv(FITZ_DIR / "manifest.csv")
    df = df[df["downloaded"] == True]  # noqa: E712
    df["image_path"] = df["filename"].apply(lambda f: FITZ_DIR / "images" / f)
    df["is_malignant"] = df["three_partition_label"] == "malignant"
    df["is_melanoma"] = df["label"].str.contains("melanoma", case=False, na=False)
    return df[["image_path", "is_malignant", "is_melanoma"]]


def main() -> None:
    all_results = []
    combined_frames = []

    for name, loader in (
        ("DDI", load_ddi), ("DDI-2", load_ddi2), ("Fitzpatrick17k (atlas)", load_fitzpatrick),
    ):
        try:
            raw = loader()
        except FileNotFoundError:
            print(f"[{name}] manifest not found, skipping")
            continue
        scored = evaluate(raw, name)
        combined_frames.append(scored.assign(source=name))
        all_results.append(report(scored, name))

    combined = pd.concat(combined_frames, ignore_index=True)
    combined_results = report(combined, "COMBINED (DDI + DDI-2 + Fitzpatrick17k)")
    all_results.append(combined_results)

    print("\n" + "=" * 68)
    print("REFERENCE: ISIC2019 test, in-domain")
    print("=" * 68)
    print("  native classifier: melanoma routed 0.7180 (confirmed reproducible)")
    print("  referral head    : melanoma routed 0.9024, AUC 0.9050")

    out_dir = REPO_ROOT / "analysis/quality/mel_sensitivity"
    pd.DataFrame(all_results).to_csv(out_dir / "domain_shift_summary.csv", index=False)
    combined.drop(columns=["image_path"]).to_csv(
        out_dir / "domain_shift_per_image.csv", index=False
    )
    print(f"\n  summary  -> {out_dir / 'domain_shift_summary.csv'}")
    print(f"  per-image -> {out_dir / 'domain_shift_per_image.csv'}")


if __name__ == "__main__":
    main()
