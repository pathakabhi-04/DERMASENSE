from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

MANIFEST_PATH = Path(
    "data/manifests/isic2019_manifest.csv"
)

OUTPUT_DIR = Path(
    "data/splits/isic2019"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ISIC 2019 LESION-LEVEL SPLIT GENERATOR")
    print("=" * 70)

    if abs(
        TRAIN_RATIO
        + VAL_RATIO
        + TEST_RATIO
        - 1.0
    ) > 1e-9:
        raise RuntimeError(
            "Split ratios must sum to 1."
        )

    df = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Loaded manifest: {len(df)} images"
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    required_columns = [
        "image",
        "native_diagnosis",
        "lesion_id",
        "lesion_id_status",
        "operational_lesion_uid",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing manifest columns: {missing}"
        )

    if df["image"].duplicated().any():
        raise RuntimeError(
            "Manifest contains duplicate images."
        )

    # --------------------------------------------------------
    # Separate identity tiers
    # --------------------------------------------------------

    identified = df[
        df["lesion_id"].notna()
    ].copy()

    unknown = df[
        df["lesion_id"].isna()
    ].copy()

    print(
        f"Identified images: {len(identified)}"
    )

    print(
        f"Unknown-ID images: {len(unknown)}"
    )

    # --------------------------------------------------------
    # Verify identified lesion consistency
    # --------------------------------------------------------

    lesion_class_counts = (
        identified
        .groupby("lesion_id")[
            "native_diagnosis"
        ]
        .nunique()
    )

    inconsistent = (
        lesion_class_counts[
            lesion_class_counts > 1
        ]
    )

    if len(inconsistent):
        raise RuntimeError(
            "Some lesion IDs contain multiple "
            "native diagnoses."
        )

    # --------------------------------------------------------
    # Build lesion-level table
    # --------------------------------------------------------

    lesions = (
        identified
        .groupby("lesion_id")
        .agg(
            native_diagnosis=(
                "native_diagnosis",
                "first",
            ),
            image_count=(
                "image",
                "count",
            ),
        )
        .reset_index()
    )

    print(
        f"Identified lesions: "
        f"{len(lesions)}"
    )

    # --------------------------------------------------------
    # Stratified lesion assignment
    # --------------------------------------------------------

    rng = np.random.default_rng(
        SEED
    )

    train_lesions = []
    val_lesions = []
    test_lesions = []

    for diagnosis, group in lesions.groupby(
        "native_diagnosis"
    ):

        class_seed = {
            "AK": 101,
            "BCC": 202,
            "BKL": 303,
            "DF": 404,
            "MEL": 505,
            "NV": 606,
            "SCC": 707,
            "VASC": 808,
        }[diagnosis]

        group = group.sample(
            frac=1.0,
            random_state=SEED + class_seed,
        ).reset_index(
            drop=True
        )

        n = len(group)

        # Initial approximate allocation.
        n_train = int(
            round(n * TRAIN_RATIO)
        )

        n_val = int(
            round(n * VAL_RATIO)
        )

        # Ensure the remainder goes to test.
        n_test = (
            n
            - n_train
            - n_val
        )

        # For classes with enough lesions, guarantee
        # at least one lesion in each split.
        if n >= 3:

            n_train = max(
                1,
                n_train
            )

            n_val = max(
                1,
                n_val
            )

            n_test = max(
                1,
                n_test
            )

            # Correct any over-allocation.
            while (
                n_train
                + n_val
                + n_test
                > n
            ):
                if n_train > n_val:
                    n_train -= 1
                elif n_val > n_test:
                    n_val -= 1
                else:
                    n_test -= 1

        train_part = group.iloc[
            :n_train
        ]

        val_part = group.iloc[
            n_train:
            n_train + n_val
        ]

        test_part = group.iloc[
            n_train + n_val:
        ]

        train_lesions.extend(
            train_part["lesion_id"]
            .tolist()
        )

        val_lesions.extend(
            val_part["lesion_id"]
            .tolist()
        )

        test_lesions.extend(
            test_part["lesion_id"]
            .tolist()
        )

    # --------------------------------------------------------
    # Sanity check lesion partition
    # --------------------------------------------------------

    train_set = set(
        train_lesions
    )

    val_set = set(
        val_lesions
    )

    test_set = set(
        test_lesions
    )

    if train_set & val_set:
        raise RuntimeError(
            "Train/Val lesion overlap."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Train/Test lesion overlap."
        )

    if val_set & test_set:
        raise RuntimeError(
            "Val/Test lesion overlap."
        )

    if (
        len(
            train_set
            | val_set
            | test_set
        )
        != len(lesions)
    ):
        raise RuntimeError(
            "Some identified lesions were "
            "not assigned to a split."
        )

    # --------------------------------------------------------
    # Map lesions → splits
    # --------------------------------------------------------

    lesion_to_split = {}

    for lesion_id in train_set:
        lesion_to_split[
            lesion_id
        ] = "train"

    for lesion_id in val_set:
        lesion_to_split[
            lesion_id
        ] = "val"

    for lesion_id in test_set:
        lesion_to_split[
            lesion_id
        ] = "test"

    identified[
        "split"
    ] = identified[
        "lesion_id"
    ].map(
        lesion_to_split
    )

    # --------------------------------------------------------
    # Unknown records → train only
    # --------------------------------------------------------

    unknown[
        "split"
    ] = "train"

    unknown[
        "identity_status"
    ] = "unknown"

    unknown[
        "evaluation_eligible"
    ] = False

    identified[
        "identity_status"
    ] = "identified"

    identified[
        "evaluation_eligible"
    ] = True

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    split_df = pd.concat(
        [
            identified,
            unknown,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Final split metadata
    # --------------------------------------------------------

    split_df[
        "split_seed"
    ] = SEED

    # Explicit evaluation flag for all rows.
    split_df[
        "evaluation_eligible"
    ] = split_df[
        "split"
    ].isin(
        [
            "val",
            "test",
        ]
    ) & (
        split_df[
            "lesion_id"
        ].notna()
    )

    # --------------------------------------------------------
    # Save split files
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        subset = split_df[
            split_df["split"]
            == split_name
        ].copy()

        subset.to_csv(
            OUTPUT_DIR
            / f"{split_name}.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows = []

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        subset = split_df[
            split_df["split"]
            == split_name
        ]

        for diagnosis, group in (
            subset.groupby(
                "native_diagnosis"
            )
        ):

            summary_rows.append(
                {
                    "split": split_name,
                    "native_diagnosis": diagnosis,
                    "images": len(group),
                    "lesions": (
                        group[
                            "lesion_id"
                        ]
                        .nunique(
                            dropna=True
                        )
                    ),
                    "unknown_identity_images": (
                        group[
                            "lesion_id"
                        ]
                        .isna()
                        .sum()
                    ),
                    "evaluation_eligible_images": (
                        group[
                            "evaluation_eligible"
                        ]
                        .sum()
                    ),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "split",
            "native_diagnosis",
        ]
    )

    summary.to_csv(
        OUTPUT_DIR
        / "split_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT STATISTICS")
    print("=" * 70)

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        subset = split_df[
            split_df["split"]
            == split_name
        ]

        print(
            f"{split_name.upper():5s}: "
            f"{len(subset):5d} images | "
            f"{subset['lesion_id'].nunique(dropna=True):5d} "
            f"identified lesions | "
            f"{subset['lesion_id'].isna().sum():4d} "
            f"unknown-ID images"
        )

    # --------------------------------------------------------
    # Explicit policy checks
    # --------------------------------------------------------

    val_unknown = (
        split_df[
            split_df["split"] == "val"
        ]["lesion_id"]
        .isna()
        .sum()
    )

    test_unknown = (
        split_df[
            split_df["split"] == "test"
        ]["lesion_id"]
        .isna()
        .sum()
    )

    if val_unknown != 0:
        raise RuntimeError(
            "Validation contains unknown-ID images."
        )

    if test_unknown != 0:
        raise RuntimeError(
            "Test contains unknown-ID images."
        )

    print(
        "\nUnknown-ID validation images: "
        f"{val_unknown}"
    )

    print(
        "Unknown-ID test images: "
        f"{test_unknown}"
    )

    print("\n" + "=" * 70)
    print("STATUS: PASS")
    print(
        "ISIC 2019 lesion-level split generated "
        "according to Split Policy v0.1."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()