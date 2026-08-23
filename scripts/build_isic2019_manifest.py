from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

GT_PATH = Path(
    "data/raw/isic2019/ground_truth/"
    "ISIC_2019_Training_GroundTruth.csv"
)

ARCHIVE_INDEX_PATH = Path(
    "data/raw/isic2019/"
    "isic2019_archive_index.csv"
)

OUTPUT_PATH = Path(
    "data/manifests/isic2019_manifest.csv"
)


# ============================================================
# NATIVE ISIC 2019 CLASSES
# ============================================================

NATIVE_CLASSES = [
    "MEL",
    "NV",
    "BCC",
    "AK",
    "BKL",
    "DF",
    "VASC",
    "SCC",
    "UNK",
]


# ============================================================
# NATIVE DIAGNOSIS
# ============================================================

def derive_native_diagnosis(row: pd.Series) -> str:
    """
    Every ISIC 2019 image should have exactly one
    positive native diagnostic class.
    """

    positives = [
        cls
        for cls in NATIVE_CLASSES
        if float(row[cls]) == 1.0
    ]

    if len(positives) != 1:
        raise RuntimeError(
            f"Expected exactly one native diagnosis for "
            f"{row['image']}, found: {positives}"
        )

    return positives[0]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING ISIC 2019 MANIFEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load validated artifacts
    # --------------------------------------------------------

    gt = pd.read_csv(GT_PATH)

    archive = pd.read_csv(
        ARCHIVE_INDEX_PATH
    )

    print(
        f"Ground truth rows: {len(gt)}"
    )

    print(
        f"Archive index rows: {len(archive)}"
    )

    # --------------------------------------------------------
    # Validate schemas
    # --------------------------------------------------------

    required_gt_columns = [
        "image",
        *NATIVE_CLASSES,
    ]

    required_archive_columns = [
        "image",
        "age_approx",
        "anatom_site_general",
        "lesion_id",
        "sex",
        "archive_path",
        "physical_filename",
        "is_downsampled",
    ]

    missing_gt = [
        c for c in required_gt_columns
        if c not in gt.columns
    ]

    missing_archive = [
        c for c in required_archive_columns
        if c not in archive.columns
    ]

    if missing_gt:
        raise RuntimeError(
            f"Ground truth missing columns: {missing_gt}"
        )

    if missing_archive:
        raise RuntimeError(
            f"Archive index missing columns: "
            f"{missing_archive}"
        )

    # --------------------------------------------------------
    # Image ID uniqueness
    # --------------------------------------------------------

    if gt["image"].duplicated().any():

        duplicates = gt[
            gt["image"].duplicated(
                keep=False
            )
        ]

        print(
            "\nDuplicate ground truth image IDs:"
        )

        print(
            duplicates[
                ["image"]
            ].to_string(index=False)
        )

        raise RuntimeError(
            "Ground truth contains duplicate image IDs."
        )

    if archive["image"].duplicated().any():

        duplicates = archive[
            archive["image"].duplicated(
                keep=False
            )
        ]

        print(
            "\nDuplicate archive image IDs:"
        )

        print(
            duplicates[
                [
                    "image",
                    "physical_filename",
                ]
            ].to_string(index=False)
        )

        raise RuntimeError(
            "Archive index contains duplicate image IDs."
        )

    # --------------------------------------------------------
    # Ground truth ↔ archive correspondence
    # --------------------------------------------------------

    gt_ids = set(
        gt["image"]
    )

    archive_ids = set(
        archive["image"]
    )

    missing_from_archive = (
        gt_ids - archive_ids
    )

    extra_in_archive = (
        archive_ids - gt_ids
    )

    print(
        f"GT images:             {len(gt_ids)}"
    )

    print(
        f"Archive images:        {len(archive_ids)}"
    )

    print(
        f"Missing from archive:  "
        f"{len(missing_from_archive)}"
    )

    print(
        f"Extra in archive:      "
        f"{len(extra_in_archive)}"
    )

    if missing_from_archive:
        print(
            "\nFirst missing images:"
        )

        print(
            sorted(
                missing_from_archive
            )[:20]
        )

        raise RuntimeError(
            "Ground truth contains images "
            "missing from archive index."
        )

    if extra_in_archive:
        print(
            "\nFirst extra images:"
        )

        print(
            sorted(
                extra_in_archive
            )[:20]
        )

        raise RuntimeError(
            "Archive index contains images "
            "not represented in ground truth."
        )

    print(
        "PASS  Ground truth ↔ archive correspondence"
    )

    # --------------------------------------------------------
    # Derive native diagnosis
    # --------------------------------------------------------

    gt["native_diagnosis"] = gt.apply(
        derive_native_diagnosis,
        axis=1
    )

    # --------------------------------------------------------
    # Merge GT with validated archive index
    # --------------------------------------------------------

    archive_columns = [
        "image",
        "age_approx",
        "anatom_site_general",
        "lesion_id",
        "sex",
        "archive_path",
        "physical_filename",
        "is_downsampled",
    ]

    manifest = gt[
        [
            "image",
            "native_diagnosis",
        ]
    ].merge(
        archive[
            archive_columns
        ],
        on="image",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Validate merge
    # --------------------------------------------------------

    if len(manifest) != len(gt):

        raise RuntimeError(
            "Manifest row count changed "
            "during merge."
        )

    if manifest[
        "archive_path"
    ].isna().any():

        raise RuntimeError(
            "Some manifest records have no "
            "physical archive mapping."
        )

    if manifest[
        "physical_filename"
    ].isna().any():

        raise RuntimeError(
            "Some manifest records have no "
            "physical filename."
        )

    # --------------------------------------------------------
    # Lesion identity status
    # --------------------------------------------------------

    manifest[
        "lesion_id_status"
    ] = (
        manifest["lesion_id"]
        .notna()
        .map(
            {
                True: "identified",
                False: "unknown",
            }
        )
    )

    # --------------------------------------------------------
    # Operational lesion UID
    #
    # IMPORTANT:
    #
    # A missing lesion_id does NOT mean that we know
    # the image is a unique lesion.
    #
    # The fallback UID is only an addressable record key.
    # It must NOT be interpreted as verified lesion identity.
    # --------------------------------------------------------

    manifest[
        "operational_lesion_uid"
    ] = (
        manifest["lesion_id"]
        .astype("string")
    )

    unknown_mask = (
        manifest["lesion_id"].isna()
    )

    manifest.loc[
        unknown_mask,
        "operational_lesion_uid"
    ] = (
        "UNKNOWN_IMAGE_"
        + manifest.loc[
            unknown_mask,
            "image"
        ].astype(str)
    )

    identified_mask = (
        manifest["lesion_id"].notna()
    )

    manifest.loc[
        identified_mask,
        "operational_lesion_uid"
    ] = (
        "LESION_"
        + manifest.loc[
            identified_mask,
            "lesion_id"
        ].astype(str)
    )

    # --------------------------------------------------------
    # Image domain
    # --------------------------------------------------------

    manifest[
        "image_domain"
    ] = "dermoscopic"

    # --------------------------------------------------------
    # Label strength
    #
    # We preserve the source label without inventing
    # additional clinical certainty categories.
    # --------------------------------------------------------

    manifest[
        "label_strength"
    ] = "native_isic2019"

    # --------------------------------------------------------
    # Final column order
    # --------------------------------------------------------

    manifest = manifest[
        [
            "image",
            "native_diagnosis",
            "lesion_id",
            "lesion_id_status",
            "operational_lesion_uid",
            "age_approx",
            "sex",
            "anatom_site_general",
            "archive_path",
            "physical_filename",
            "is_downsampled",
            "image_domain",
            "label_strength",
        ]
    ]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        "\nManifest statistics"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Images:                 "
        f"{len(manifest)}"
    )

    print(
        f"Unique images:          "
        f"{manifest['image'].nunique()}"
    )

    print(
        f"Unique source lesions:  "
        f"{manifest['lesion_id'].nunique(dropna=True)}"
    )

    print(
        f"Images with lesion ID:  "
        f"{manifest['lesion_id'].notna().sum()}"
    )

    print(
        f"Images without lesion:  "
        f"{manifest['lesion_id'].isna().sum()}"
    )

    # --------------------------------------------------------
    # Native diagnosis counts
    # --------------------------------------------------------

    print(
        "\nNative diagnoses"
    )

    print(
        manifest[
            "native_diagnosis"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Lesion ID status
    # --------------------------------------------------------

    print(
        "\nLesion ID status"
    )

    print(
        manifest[
            "lesion_id_status"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Image domain
    # --------------------------------------------------------

    print(
        "\nImage domain"
    )

    print(
        manifest[
            "image_domain"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved manifest:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "STATUS: PASS"
    )

    print(
        "ISIC 2019 manifest built successfully "
        "from validated ground truth and archive index."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()