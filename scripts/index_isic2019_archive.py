from pathlib import Path
import zipfile

import pandas as pd


ZIP_PATH = Path(
    "data/raw/isic2019/ISIC_2019_Training_Input.zip"
)

GT_PATH = Path(
    "data/raw/isic2019/ground_truth/ISIC_2019_Training_GroundTruth.csv"
)

META_PATH = Path(
    "data/raw/isic2019/metadata/ISIC_2019_Training_Metadata.csv"
)

OUTPUT_PATH = Path(
    "data/raw/isic2019/isic2019_archive_index.csv"
)

EXPECTED_IMAGE_COUNT = 25331


def main():

    print("=" * 70)
    print("ISIC 2019 ARCHIVE INDEX BUILDER")
    print("=" * 70)

    # ---------------------------------------------------------
    # Check required files
    # ---------------------------------------------------------

    for path in [ZIP_PATH, GT_PATH, META_PATH]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # ---------------------------------------------------------
    # Load CSV files
    # ---------------------------------------------------------

    gt = pd.read_csv(GT_PATH)
    meta = pd.read_csv(META_PATH)

    gt_ids = set(gt["image"])
    meta_ids = set(meta["image"])

    print(
        f"Ground truth rows: {len(gt)}"
    )

    print(
        f"Metadata rows:     {len(meta)}"
    )

    print(
        f"CSV image IDs:     {len(gt_ids)}"
    )

    # ---------------------------------------------------------
    # GT ↔ metadata ID validation
    # ---------------------------------------------------------

    if gt_ids != meta_ids:

        gt_only = gt_ids - meta_ids
        meta_only = meta_ids - gt_ids

        print(
            f"GT-only IDs:       {len(gt_only)}"
        )

        print(
            f"Metadata-only IDs: {len(meta_only)}"
        )

        raise RuntimeError(
            "Ground truth and metadata image IDs do not match."
        )

    if len(gt_ids) != EXPECTED_IMAGE_COUNT:

        raise RuntimeError(
            f"Expected {EXPECTED_IMAGE_COUNT} image IDs, "
            f"found {len(gt_ids)}."
        )

    print(
        "PASS  Ground truth ↔ metadata IDs"
    )

    # ---------------------------------------------------------
    # Read archive
    # ---------------------------------------------------------

    records = []

    with zipfile.ZipFile(ZIP_PATH, "r") as z:

        for name in z.namelist():

            # Only JPEG images.
            if not name.lower().endswith(".jpg"):
                continue

            filename = Path(name).name

            # Ignore documentation files.
            if not filename.startswith("ISIC_"):
                continue

            # -------------------------------------------------
            # IMPORTANT:
            #
            # The CSV uses the complete filename stem as its
            # image identifier.
            #
            # Therefore:
            #
            # ISIC_0000017.jpg
            #     -> ISIC_0000017
            #
            # ISIC_0000017_downsampled.jpg
            #     -> ISIC_0000017_downsampled
            #
            # Do NOT remove "_downsampled".
            # -------------------------------------------------

            image_id = Path(filename).stem

            records.append(
                {
                    "image_id": image_id,
                    "archive_path": name,
                    "physical_filename": filename,
                    "is_downsampled": image_id.endswith(
                        "_downsampled"
                    ),
                }
            )

    # ---------------------------------------------------------
    # Build archive DataFrame
    # ---------------------------------------------------------

    archive_df = pd.DataFrame(records)

    print(
        f"JPEG records found: {len(archive_df)}"
    )

    if len(archive_df) != EXPECTED_IMAGE_COUNT:

        raise RuntimeError(
            f"Expected {EXPECTED_IMAGE_COUNT} JPEGs, "
            f"found {len(archive_df)}."
        )

    # ---------------------------------------------------------
    # Duplicate archive IDs
    # ---------------------------------------------------------

    duplicates = archive_df[
        archive_df["image_id"].duplicated(keep=False)
    ]

    if len(duplicates):

        print(
            "\nERROR: Duplicate archive image IDs:"
        )

        print(
            duplicates.to_string(index=False)
        )

        raise RuntimeError(
            "Duplicate physical image IDs detected."
        )

    print(
        "PASS  Archive image IDs are unique"
    )

    # ---------------------------------------------------------
    # Archive ↔ CSV correspondence
    # ---------------------------------------------------------

    archive_ids = set(
        archive_df["image_id"]
    )

    missing_from_archive = (
        gt_ids - archive_ids
    )

    extra_in_archive = (
        archive_ids - gt_ids
    )

    print(
        f"CSV IDs missing from archive: "
        f"{len(missing_from_archive)}"
    )

    print(
        f"Archive IDs missing from CSV: "
        f"{len(extra_in_archive)}"
    )

    if missing_from_archive:

        print(
            "\nFirst missing IDs:"
        )

        print(
            sorted(missing_from_archive)[:20]
        )

        raise RuntimeError(
            "CSV contains images missing from archive."
        )

    if extra_in_archive:

        print(
            "\nFirst extra IDs:"
        )

        print(
            sorted(extra_in_archive)[:20]
        )

        raise RuntimeError(
            "Archive contains images missing from CSV."
        )

    print(
        "PASS  Archive ↔ CSV correspondence"
    )

    # ---------------------------------------------------------
    # Filename statistics
    # ---------------------------------------------------------

    downsampled_count = int(
        archive_df["is_downsampled"].sum()
    )

    standard_count = (
        len(archive_df) - downsampled_count
    )

    print(
        "\nPhysical filename statistics"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Total JPEGs:       {len(archive_df)}"
    )

    print(
        f"Downsampled:       {downsampled_count}"
    )

    print(
        f"Standard filename: {standard_count}"
    )

    # ---------------------------------------------------------
    # Metadata ↔ archive merge
    # ---------------------------------------------------------

    manifest_preview = meta.merge(
        archive_df,
        left_on="image",
        right_on="image_id",
        how="left",
        validate="one_to_one",
    )

    missing_paths = (
        manifest_preview["archive_path"].isna()
    )

    if missing_paths.any():

        count = int(
            missing_paths.sum()
        )

        raise RuntimeError(
            f"{count} metadata records failed "
            "archive mapping."
        )

    if len(manifest_preview) != len(meta):

        raise RuntimeError(
            "Archive merge changed metadata row count."
        )

    manifest_preview = manifest_preview.drop(
        columns=["image_id"]
    )

    print(
        "PASS  Metadata ↔ archive mapping"
    )

    # ---------------------------------------------------------
    # Save index
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_preview.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nSaved archive index:"
    )

    print(
        OUTPUT_PATH
    )

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "STATUS: PASS"
    )

    print(
        "All 25,331 CSV image IDs map to exactly one "
        "physical JPEG in the archive."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
