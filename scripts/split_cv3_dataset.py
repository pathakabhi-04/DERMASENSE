from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


MANIFEST = Path(
    "data/manifests/isic2018_task1_manifest.csv"
)

OUTPUT_DIR = Path(
    "data/splits/isic2018_task1"
)

SEED = 42


def main() -> None:
    df = pd.read_csv(MANIFEST)

    df = df[
        df["cv3_eligible"] == True
    ].copy()

    if len(df) != 2593:
        raise RuntimeError(
            f"Expected 2593 eligible images, got {len(df)}"
        )

    if df["image_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate image IDs detected."
        )

    # Deterministic 80/10/10 split.
    train, temp = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        shuffle=True,
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=SEED,
        shuffle=True,
    )

    splits = {
        "train": train,
        "val": val,
        "test": test,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, split in splits.items():
        split = split.sort_values(
            "image_id"
        ).reset_index(drop=True)

        split.to_csv(
            OUTPUT_DIR / f"{name}.csv",
            index=False,
        )

    summary = []

    for name, split in splits.items():
        summary.append(
            {
                "split": name,
                "images": len(split),
                "overlap_isic2019_train": int(
                    (
                        split["isic2019_split"]
                        == "train"
                    ).sum()
                ),
                "overlap_isic2019_val": int(
                    (
                        split["isic2019_split"]
                        == "val"
                    ).sum()
                ),
                "overlap_isic2019_test": int(
                    (
                        split["isic2019_split"]
                        == "test"
                    ).sum()
                ),
            }
        )

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        OUTPUT_DIR / "split_summary.csv",
        index=False,
    )

    # Hard assertions.
    train_ids = set(train["image_id"])
    val_ids = set(val["image_id"])
    test_ids = set(test["image_id"])

    assert not train_ids & val_ids
    assert not train_ids & test_ids
    assert not val_ids & test_ids

    assert (
        len(train_ids | val_ids | test_ids)
        == 2593
    )

    print("=" * 80)
    print("DERMASENSE CV-2 DATASET SPLIT")
    print("=" * 80)

    print()
    print(summary_df.to_string(index=False))

    print()
    print("Leakage checks:")
    print(
        "Train ∩ Val: ",
        len(train_ids & val_ids),
    )
    print(
        "Train ∩ Test:",
        len(train_ids & test_ids),
    )
    print(
        "Val ∩ Test:  ",
        len(val_ids & test_ids),
    )

    print()
    print(
        "Total assigned:",
        len(train_ids | val_ids | test_ids),
    )

    print()
    print(
        f"Saved: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
