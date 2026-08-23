from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

MANIFEST_PATH = Path(
    "data/manifests/isic2019_manifest.csv"
)

SPLIT_DIR = Path(
    "data/splits/isic2019"
)


# ============================================================
# EXPECTED VALUES FROM RAW AUDIT
# ============================================================

EXPECTED_IMAGES = 25331
EXPECTED_IDENTIFIED_IMAGES = 23247
EXPECTED_UNKNOWN_IMAGES = 2084
EXPECTED_LESIONS = 11847

VALID_DIAGNOSES = {
    "AK",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "SCC",
    "VASC",
}

SPLITS = [
    "train",
    "val",
    "test",
]


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print(f"FAIL  {message}")
    raise RuntimeError(message)


def check(condition, message):
    if condition:
        print(f"PASS  {message}")
    else:
        fail(message)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ISIC 2019 INDEPENDENT SPLIT VALIDATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():
        fail(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    for split_name in SPLITS:
        path = (
            SPLIT_DIR
            / f"{split_name}.csv"
        )

        if not path.exists():
            fail(
                f"Split file not found: {path}"
            )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    split_dfs = {
        split_name: pd.read_csv(
            SPLIT_DIR
            / f"{split_name}.csv"
        )
        for split_name in SPLITS
    }

    print("\nFiles loaded successfully.")

    print(
        f"Manifest rows: "
        f"{len(manifest)}"
    )

    for split_name in SPLITS:
        print(
            f"{split_name.capitalize():<6} rows: "
            f"{len(split_dfs[split_name])}"
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_manifest_columns = {
        "image",
        "native_diagnosis",
        "lesion_id",
    }

    required_split_columns = {
        "image",
        "native_diagnosis",
        "lesion_id",
        "split",
        "identity_status",
        "evaluation_eligible",
    }

    check(
        required_manifest_columns
        .issubset(manifest.columns),
        "Manifest contains required columns",
    )

    for split_name, df in split_dfs.items():

        check(
            required_split_columns
            .issubset(df.columns),
            f"{split_name} contains required columns",
        )

    # ========================================================
    # MANIFEST INTEGRITY
    # ========================================================

    print("\n" + "-" * 70)
    print("MANIFEST INTEGRITY")
    print("-" * 70)

    check(
        not manifest["image"].duplicated().any(),
        "Manifest image IDs are unique",
    )

    check(
        len(manifest) == EXPECTED_IMAGES,
        f"Manifest contains {EXPECTED_IMAGES} images",
    )

    identified_manifest = manifest[
        manifest["lesion_id"].notna()
    ]

    unknown_manifest = manifest[
        manifest["lesion_id"].isna()
    ]

    check(
        len(identified_manifest)
        == EXPECTED_IDENTIFIED_IMAGES,
        f"Manifest contains "
        f"{EXPECTED_IDENTIFIED_IMAGES} identified images",
    )

    check(
        len(unknown_manifest)
        == EXPECTED_UNKNOWN_IMAGES,
        f"Manifest contains "
        f"{EXPECTED_UNKNOWN_IMAGES} unknown-ID images",
    )

    check(
        identified_manifest[
            "lesion_id"
        ].nunique()
        == EXPECTED_LESIONS,
        f"Manifest contains "
        f"{EXPECTED_LESIONS} identified lesions",
    )

    check(
        set(
            manifest["native_diagnosis"]
        ).issubset(
            VALID_DIAGNOSES
        ),
        "Manifest native diagnoses are valid",
    )

    # ========================================================
    # INDIVIDUAL SPLIT VALIDATION
    # ========================================================

    print("\n" + "-" * 70)
    print("INDIVIDUAL SPLIT VALIDATION")
    print("-" * 70)

    manifest_image_ids = set(
        manifest["image"]
    )

    for split_name in SPLITS:

        df = split_dfs[
            split_name
        ]

        print(
            f"\n[{split_name.upper()}]"
        )

        # ----------------------------------------------------
        # Image uniqueness
        # ----------------------------------------------------

        check(
            not df["image"]
            .duplicated()
            .any(),
            "No duplicate image IDs",
        )

        # ----------------------------------------------------
        # Image existence
        # ----------------------------------------------------

        unknown_images = (
            set(df["image"])
            - manifest_image_ids
        )

        check(
            len(unknown_images) == 0,
            "All image IDs exist in manifest",
        )

        # ----------------------------------------------------
        # Diagnosis validity
        # ----------------------------------------------------

        check(
            set(
                df["native_diagnosis"]
            ).issubset(
                VALID_DIAGNOSES
            ),
            "Native diagnoses valid",
        )

        # ----------------------------------------------------
        # Split column integrity
        # ----------------------------------------------------

        check(
            set(
                df["split"].dropna()
            ) == {split_name},
            f"All rows correctly marked as {split_name}",
        )

        # ----------------------------------------------------
        # Identity status
        # ----------------------------------------------------

        identified_rows = df[
            df["lesion_id"].notna()
        ]

        unknown_rows = df[
            df["lesion_id"].isna()
        ]

        if split_name in {
            "val",
            "test",
        }:

            check(
                len(unknown_rows) == 0,
                "No unknown-lesion-ID images in evaluation split",
            )

            check(
                (
                    df[
                        "evaluation_eligible"
                    ]
                    == True
                ).all(),
                "All evaluation images are marked eligible",
            )

        else:

            # Training may contain unknown identities.
            check(
                (
                    unknown_rows[
                        "evaluation_eligible"
                    ]
                    == False
                ).all(),
                "Unknown-ID training images are not evaluation eligible",
            )

        # ----------------------------------------------------
        # Lesion UID validation
        # ----------------------------------------------------

        if len(identified_rows):

            lesion_ids = set(
                identified_rows[
                    "lesion_id"
                ]
            )

            manifest_lesions = set(
                identified_manifest[
                    "lesion_id"
                ]
            )

            check(
                lesion_ids
                .issubset(
                    manifest_lesions
                ),
                "All lesion IDs exist in manifest",
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        print(
            f"Images:   {len(df)}"
        )

        print(
            f"Lesions:  "
            f"{df['lesion_id'].nunique(dropna=True)}"
        )

        print(
            f"Unknown:  "
            f"{df['lesion_id'].isna().sum()}"
        )

        print(
            f"Eligible: "
            f"{df['evaluation_eligible'].sum()}"
        )

        print("\nDiagnosis counts:")

        print(
            df[
                "native_diagnosis"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    # ========================================================
    # GLOBAL IMAGE PARTITION
    # ========================================================

    print("\n" + "-" * 70)
    print("GLOBAL PARTITION CHECK")
    print("-" * 70)

    train_images = set(
        split_dfs["train"]["image"]
    )

    val_images = set(
        split_dfs["val"]["image"]
    )

    test_images = set(
        split_dfs["test"]["image"]
    )

    check(
        len(
            train_images
            & val_images
        ) == 0,
        "Train ↔ Val image overlap: 0",
    )

    check(
        len(
            train_images
            & test_images
        ) == 0,
        "Train ↔ Test image overlap: 0",
    )

    check(
        len(
            val_images
            & test_images
        ) == 0,
        "Val ↔ Test image overlap: 0",
    )

    all_split_images = (
        train_images
        | val_images
        | test_images
    )

    check(
        all_split_images
        == manifest_image_ids,
        "Every manifest image appears exactly once",
    )

    check(
        len(all_split_images)
        == EXPECTED_IMAGES,
        f"Exactly {EXPECTED_IMAGES} unique images across splits",
    )

    # ========================================================
    # IDENTIFIED LESION PARTITION
    # ========================================================

    print("\n" + "-" * 70)
    print("IDENTIFIED LESION PARTITION")
    print("-" * 70)

    train_lesions = set(
        split_dfs["train"]
        .loc[
            split_dfs["train"]["lesion_id"].notna(),
            "lesion_id",
        ]
    )

    val_lesions = set(
        split_dfs["val"]
        .loc[
            split_dfs["val"]["lesion_id"].notna(),
            "lesion_id",
        ]
    )

    test_lesions = set(
        split_dfs["test"]
        .loc[
            split_dfs["test"]["lesion_id"].notna(),
            "lesion_id",
        ]
    )

    check(
        len(
            train_lesions
            & val_lesions
        ) == 0,
        "Train ↔ Val lesion overlap: 0",
    )

    check(
        len(
            train_lesions
            & test_lesions
        ) == 0,
        "Train ↔ Test lesion overlap: 0",
    )

    check(
        len(
            val_lesions
            & test_lesions
        ) == 0,
        "Val ↔ Test lesion overlap: 0",
    )

    all_split_lesions = (
        train_lesions
        | val_lesions
        | test_lesions
    )

    check(
        all_split_lesions
        == set(
            identified_manifest[
                "lesion_id"
            ]
        ),
        "All identified lesions are partitioned exactly once",
    )

    check(
        len(all_split_lesions)
        == EXPECTED_LESIONS,
        f"Exactly {EXPECTED_LESIONS} identified lesions across splits",
    )

    # ========================================================
    # UNKNOWN-ID POLICY
    # ========================================================

    print("\n" + "-" * 70)
    print("UNKNOWN LESION-ID POLICY")
    print("-" * 70)

    train_unknown = split_dfs[
        "train"
    ]["lesion_id"].isna().sum()

    val_unknown = split_dfs[
        "val"
    ]["lesion_id"].isna().sum()

    test_unknown = split_dfs[
        "test"
    ]["lesion_id"].isna().sum()

    check(
        train_unknown
        == EXPECTED_UNKNOWN_IMAGES,
        f"All {EXPECTED_UNKNOWN_IMAGES} unknown-ID images are in train",
    )

    check(
        val_unknown == 0,
        "Validation contains zero unknown-ID images",
    )

    check(
        test_unknown == 0,
        "Test contains zero unknown-ID images",
    )

    # ========================================================
    # EVALUATION ELIGIBILITY
    # ========================================================

    print("\n" + "-" * 70)
    print("EVALUATION ELIGIBILITY")
    print("-" * 70)

    val = split_dfs["val"]
    test = split_dfs["test"]

    check(
        val["evaluation_eligible"].all(),
        "All validation images are evaluation eligible",
    )

    check(
        test["evaluation_eligible"].all(),
        "All test images are evaluation eligible",
    )

    check(
        not split_dfs["train"]
        .loc[
            split_dfs["train"]["lesion_id"].isna(),
            "evaluation_eligible",
        ]
        .any(),
        "Unknown-ID training images are evaluation ineligible",
    )

    # ========================================================
    # CLASS COVERAGE
    # ========================================================

    print("\n" + "-" * 70)
    print("CLASS COVERAGE")
    print("-" * 70)

    class_table = pd.DataFrame(
        {
            split_name: (
                split_dfs[split_name][
                    "native_diagnosis"
                ]
                .value_counts()
                .reindex(
                    sorted(
                        VALID_DIAGNOSES
                    ),
                    fill_value=0,
                )
            )
            for split_name in SPLITS
        }
    )

    class_table = class_table[
        SPLITS
    ]

    print(
        class_table.to_string()
    )

    for diagnosis in sorted(
        VALID_DIAGNOSES
    ):

        for split_name in SPLITS:

            count = int(
                class_table.loc[
                    diagnosis,
                    split_name,
                ]
            )

            if count == 0:

                # We do not automatically fail here because
                # the policy says "sufficient identified-lesion
                # support". The validator reports the issue.
                print(
                    f"WARNING  {diagnosis} has "
                    f"zero images in {split_name}"
                )

    # ========================================================
    # PATIENT-LEVEL LIMITATION
    # ========================================================

    print("\n" + "-" * 70)
    print("PATIENT-LEVEL GUARANTEE")
    print("-" * 70)

    if "patient_id" not in manifest.columns:

        print(
            "INFO  patient_id is absent from ISIC 2019 metadata."
        )

        print(
            "INFO  Patient-level leakage cannot be "
            "mechanically verified."
        )

    else:

        fail(
            "Unexpected patient_id column found. "
            "Review split policy before proceeding."
        )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 70)
    print("STATUS: PASS")
    print("=" * 70)

    print(
        "Independent validation confirms that the "
        "ISIC 2019 split artifacts satisfy "
        "Split Policy v0.1."
    )

    print(
        "Evaluation sets are lesion-disjoint and "
        "contain no unknown-lesion-ID images."
    )

    print(
        "Patient-level independence is NOT guaranteed "
        "because patient_id is unavailable."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()