"""
Leave-one-clinical-domain-out splits for the multi-domain run
(docs/cv4b_domain_generalization_design.md §5).

Unlike `build_cv4b_splits.py`, every fold column here is authoritative
for EVERY image -- ISIC and PAD-UFES included -- so the training script
needs no per-source special-casing at all.

    fold           trains on                    tests on
    dg_atlas       isic + pad + stanford        Fitzpatrick17k
    dg_stanford    isic + pad + atlas           DDI + DDI-2
    dg_pad         isic + stanford + atlas      PAD-UFES test

Rules, applied per fold:

  ISIC          always its own frozen split (train/val/test). Its `test`
                rows stay `test` so the no-forgetting floor can be
                measured; the evaluator separates ISIC from the held-out
                clinical domain by source.
  PAD-UFES      held out -> its frozen TEST split becomes the fold's
                clinical test set and its train/val become `unused`.
                Otherwise its frozen train/val are used and its test is
                `unused` (a fold has exactly one held-out domain).
  stanford      held out -> all of it is `test`. Otherwise 75/25
  / atlas       train/val, stratified and patient-grouped exactly as
                `build_cv4b_splits.py` does.

`unused` is explicit rather than blank so a row is never silently
dropped by a filter that meant to include it.

    PYTHONPATH=. python3 scripts/build_dg_splits.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
CLINICAL_SPLITS = REPO_ROOT / "analysis/quality/mel_sensitivity/cv4b_finetune_splits.csv"
OUT = REPO_ROOT / "analysis/quality/mel_sensitivity/cv4b_dg_splits.csv"

SEED = 20260914
REFER = ("MEL", "BCC", "SCC", "ACK")   # PAD-UFES label space
BENIGN = ("NEV", "SEK")
FOLDS = {"dg_atlas": "atlas", "dg_stanford": "stanford", "dg_pad": "pad"}


def pad_rows() -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        table = pd.read_csv(REPO_ROOT / f"data/splits/pad_ufes/{split}.csv")
        diagnosis = table["native_diagnosis"].str.upper()
        keep = diagnosis.isin(REFER + BENIGN)
        frames.append(pd.DataFrame({
            "image_path": table.loc[keep, "image_path"].map(lambda p: str(REPO_ROOT / p)),
            "source": "pad_ufes",
            "domain_group": "pad",
            "group": table.loc[keep, "patient_id"].astype(str),  # real patient IDs
            "is_malignant": diagnosis[keep].isin(REFER),
            "is_melanoma": diagnosis[keep] == "MEL",
            "native_split": split,
        }))
    return pd.concat(frames, ignore_index=True)


def clinical_rows() -> pd.DataFrame:
    table = pd.read_csv(CLINICAL_SPLITS)
    return pd.DataFrame({
        "image_path": table["image_path"],
        "source": table["source"],
        "domain_group": table["source_group"],   # stanford | atlas
        "group": table["group"],
        "is_malignant": table["is_malignant"],
        "is_melanoma": table["is_melanoma"],
        "native_split": "",
    })


def train_val_split(frame: pd.DataFrame) -> pd.Series:
    """75/25, stratified by (source, is_malignant), patient-grouped."""
    strata = frame["source"] + "_" + frame["is_malignant"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    _, val_index = next(splitter.split(frame, strata, groups=frame["group"]))
    assignment = pd.Series("train", index=frame.index)
    assignment.iloc[val_index] = "val"
    return assignment


def assign_fold(data: pd.DataFrame, held_out: str) -> pd.Series:
    fold = pd.Series("unused", index=data.index)

    is_isic = data["domain_group"] == "isic"
    fold[is_isic] = data.loc[is_isic, "native_split"]

    is_pad = data["domain_group"] == "pad"
    if held_out == "pad":
        fold[is_pad & (data["native_split"] == "test")] = "test"
    else:
        in_training = is_pad & data["native_split"].isin(["train", "val"])
        fold[in_training] = data.loc[in_training, "native_split"]

    for group in ("stanford", "atlas"):
        rows = data["domain_group"] == group
        if group == held_out:
            fold[rows] = "test"
        else:
            subset = data[rows]
            fold.loc[subset.index] = train_val_split(subset)

    return fold


def isic_placeholder() -> pd.DataFrame:
    """ISIC rows are added by the bundle builder, which already resolves
    their paths and labels. This script only needs to know they exist so
    the fold columns can be written for them there."""
    return pd.DataFrame(columns=[
        "image_path", "source", "domain_group", "group",
        "is_malignant", "is_melanoma", "native_split",
    ])


def main() -> None:
    data = pd.concat([pad_rows(), clinical_rows(), isic_placeholder()], ignore_index=True)
    print(f"{len(data)} non-ISIC images "
          f"({data['domain_group'].value_counts().to_dict()})")

    for fold, held_out in FOLDS.items():
        data[fold] = assign_fold(data, held_out)
        spans = data[data[fold] != "unused"].groupby("group")[fold].nunique()
        if (spans > 1).any():
            raise RuntimeError(f"{fold}: {(spans > 1).sum()} groups span two splits")

    print()
    for fold in FOLDS:
        print(f"[{fold}]")
        summary = data[data[fold] != "unused"].groupby([fold, "domain_group"]).agg(
            n=("is_malignant", "size"),
            malignant=("is_malignant", "sum"),
            melanoma=("is_melanoma", "sum"),
        )
        print(summary.to_string())
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT, index=False)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
