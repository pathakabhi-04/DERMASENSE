"""
Build the splits for the CV-4b backbone fine-tune, fixing a real defect in
`cv4b_retrain_split.csv` (the linear-probe refit's split) and adding the
leave-one-source-out folds the GPU run needs.

## The defect being fixed

`cv4b_retrain_split.csv` split DDI-2 by IMAGE. DDI-2 carries
`deidentified_patient_id`, and 89 of its 550 patients contribute more
than one photo (204 images total). 25 of that split's 132 held-out DDI-2
test images come from patients who also appear in its training portion.
That is patient-level leakage: two photos of the same patient's lesion
are not independent samples, and a model can score the test one by
having memorised the train one.

Nothing here is grouped by lesion, only by patient, because that is the
only grouping key any of these datasets actually provide.

## Grouping, honestly stated

  - DDI-2  : grouped by `deidentified_patient_id` (real).
  - DDI    : no patient identifier exists in its manifest. Each image is
             treated as its own group. If DDI in fact contains repeat
             patients, this split cannot detect or prevent it.
  - Fitz17k: same -- no patient identifier. Atlas teaching images, one
             group per image.

DDI and DDI-2 are BOTH Stanford AIMI releases. Cross-dataset patient
overlap between them cannot be ruled out (DDI has no patient IDs to
check against), so for leave-one-source-out they are treated as ONE
source group ("stanford"), never split against each other. Holding out
DDI while training on DDI-2 would not be an unseen-source test.

## Splits produced

  pooled           : 60/20/20 train/val/test, stratified by
                     (source, is_malignant), grouped as above. The
                     in-distribution measurement.
  loso_atlas       : test = Fitzpatrick17k entirely; train/val from
                     stanford. "Does adaptation reach an unseen source?"
  loso_stanford    : test = DDI + DDI-2 entirely; train/val from atlas.
                     The same question, other direction.

Deployment is a fourth, unseen source (a user's phone), so the LOSO
folds are the honest proxy for it and `pooled` is the optimistic one.
Both are reported.

    PYTHONPATH=. python3 scripts/build_cv4b_splits.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from scripts.run_domain_shift_check import load_ddi, load_ddi2, load_fitzpatrick

REPO_ROOT = Path(__file__).resolve().parents[1]
DDI2_MANIFEST = REPO_ROOT / "data/raw/ddi2/manifest.csv"
OUT = REPO_ROOT / "analysis/quality/mel_sensitivity/cv4b_finetune_splits.csv"

SEED = 20260913
SOURCE_GROUP = {"ddi": "stanford", "ddi2": "stanford", "fitzpatrick17k": "atlas"}


def build_frame() -> pd.DataFrame:
    frames = []
    for name, loader in (
        ("ddi", load_ddi), ("ddi2", load_ddi2), ("fitzpatrick17k", load_fitzpatrick),
    ):
        df = loader().copy()
        df["source"] = name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["image_path"] = data["image_path"].astype(str)
    data["source_group"] = data["source"].map(SOURCE_GROUP)

    # Patient grouping: real for DDI-2, per-image everywhere else.
    ddi2 = pd.read_csv(DDI2_MANIFEST)
    ddi2["photo_id"] = ddi2["photo_id"].astype(str)
    patient_by_photo = dict(zip(ddi2["photo_id"], ddi2["deidentified_patient_id"].astype(str)))

    def group_key(row) -> str:
        if row["source"] == "ddi2":
            patient = patient_by_photo.get(Path(row["image_path"]).stem)
            if patient is None:
                raise RuntimeError(
                    f"DDI-2 image {row['image_path']} has no patient id in the manifest; "
                    "refusing to guess -- that is exactly the leak this script exists to stop."
                )
            return f"ddi2_patient_{patient}"
        return f"{row['source']}_image_{Path(row['image_path']).stem}"

    data["group"] = data.apply(group_key, axis=1)
    return data


def stratified_group_split(data: pd.DataFrame) -> pd.Series:
    """60/20/20 via 5 stratified group folds: 3 train, 1 val, 1 test."""
    strata = data["source"] + "_" + data["is_malignant"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.empty(len(data), dtype=int)
    for fold_index, (_, held_out) in enumerate(
        splitter.split(data, strata, groups=data["group"])
    ):
        folds[held_out] = fold_index
    return pd.Series(
        np.select([folds == 0, folds == 1], ["test", "val"], default="train"),
        index=data.index,
    )


def loso_split(data: pd.DataFrame, holdout_group: str) -> pd.Series:
    """Held-out source group is entirely test; the rest splits 75/25 train/val."""
    split = pd.Series("train", index=data.index)
    held = data["source_group"] == holdout_group
    split[held] = "test"

    trainable = data[~held]
    strata = trainable["source"] + "_" + trainable["is_malignant"].astype(str)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    _, val_index = next(splitter.split(trainable, strata, groups=trainable["group"]))
    split.loc[trainable.index[val_index]] = "val"
    return split


def assert_no_leakage(data: pd.DataFrame, column: str) -> None:
    spans = data.groupby("group")[column].nunique()
    offenders = spans[spans > 1]
    if len(offenders):
        raise RuntimeError(
            f"{column}: {len(offenders)} group(s) span more than one split -- "
            "this is the leak the script exists to prevent."
        )
    overlap = data.groupby("image_path")[column].nunique()
    if (overlap > 1).any():
        raise RuntimeError(f"{column}: an image appears in more than one split.")


def main() -> None:
    data = build_frame()
    print(f"{len(data)} images, {data['group'].nunique()} groups "
          f"({(data['source'] == 'ddi2').sum()} DDI-2 images in "
          f"{data[data['source'] == 'ddi2']['group'].nunique()} patients)")

    data["pooled"] = stratified_group_split(data)
    data["loso_atlas"] = loso_split(data, "atlas")
    data["loso_stanford"] = loso_split(data, "stanford")

    for column in ("pooled", "loso_atlas", "loso_stanford"):
        assert_no_leakage(data, column)
        print(f"\n[{column}] no group or image spans two splits")
        summary = data.groupby([column, "source"]).agg(
            n=("is_malignant", "size"),
            malignant=("is_malignant", "sum"),
            melanoma=("is_melanoma", "sum"),
        )
        print(summary)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    data[[
        "image_path", "source", "source_group", "group",
        "is_malignant", "is_melanoma", "pooled", "loso_atlas", "loso_stanford",
    ]].to_csv(OUT, index=False)
    print(f"\nsplits -> {OUT}")


if __name__ == "__main__":
    main()
