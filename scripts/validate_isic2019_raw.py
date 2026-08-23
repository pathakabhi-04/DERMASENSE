from pathlib import Path

import pandas as pd
from PIL import Image


BASE_DIR = Path("data/raw/isic2019")

IMAGE_DIR = (
    BASE_DIR / "ISIC_2019_Training_Input"
)

GT_PATH = (
    BASE_DIR
    / "ground_truth"
    / "ISIC_2019_Training_GroundTruth.csv"
)

META_PATH = (
    BASE_DIR
    / "metadata"
    / "ISIC_2019_Training_Metadata.csv"
)

EXPECTED_IMAGES = 25331


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


def main():

    print("=" * 70)
    print("ISIC 2019 RAW DATA VALIDATION")
    print("=" * 70)

    # =========================================================
    # 1. Check files
    # =========================================================

    for path in [IMAGE_DIR, GT_PATH, META_PATH]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found: {path}"
            )

    # =========================================================
    # 2. Load metadata and ground truth
    # =========================================================

    gt = pd.read_csv(GT_PATH)
    meta = pd.read_csv(META_PATH)

    print("\n[1] Metadata / Ground Truth")

    print(
        f"Ground truth rows: {len(gt)}"
    )

    print(
        f"Metadata rows:     {len(meta)}"
    )

    print(
        f"GT unique images:   {gt['image'].nunique()}"
    )

    print(
        f"Metadata unique:    {meta['image'].nunique()}"
    )

    # =========================================================
    # 3. Basic row validation
    # =========================================================

    if len(gt) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} GT rows, "
            f"found {len(gt)}"
        )

    if len(meta) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} metadata rows, "
            f"found {len(meta)}"
        )

    if gt["image"].nunique() != EXPECTED_IMAGES:
        raise RuntimeError(
            "Duplicate image IDs found in ground truth."
        )

    if meta["image"].nunique() != EXPECTED_IMAGES:
        raise RuntimeError(
            "Duplicate image IDs found in metadata."
        )

    print(
        "PASS  Ground truth and metadata row counts"
    )

    print(
        "PASS  Ground truth image IDs unique"
    )

    print(
        "PASS  Metadata image IDs unique"
    )

    # =========================================================
    # 4. Native diagnosis validation
    # =========================================================

    print("\n[2] Native Diagnostic Counts")

    label_sum = gt[NATIVE_CLASSES].sum(axis=1)

    zero_labels = int(
        (label_sum == 0).sum()
    )

    multiple_labels = int(
        (label_sum > 1).sum()
    )

    print(
        f"Images with zero labels:      {zero_labels}"
    )

    print(
        f"Images with multiple labels:  {multiple_labels}"
    )

    if zero_labels != 0:
        raise RuntimeError(
            "Images without a native diagnosis found."
        )

    if multiple_labels != 0:
        raise RuntimeError(
            "Images with multiple native diagnoses found."
        )

    for cls in NATIVE_CLASSES:

        count = int(
            (gt[cls] == 1).sum()
        )

        print(
            f"{cls:5s}: {count:6d}"
        )

    print(
        "PASS  Every image has exactly one native diagnosis"
    )

    # =========================================================
    # 5. GT ↔ metadata correspondence
    # =========================================================

    print(
        "\n[3] Ground Truth ↔ Metadata Correspondence"
    )

    gt_ids = set(gt["image"])
    meta_ids = set(meta["image"])

    missing_metadata = gt_ids - meta_ids
    extra_metadata = meta_ids - gt_ids

    print(
        f"GT IDs missing from metadata: "
        f"{len(missing_metadata)}"
    )

    print(
        f"Metadata IDs missing from GT: "
        f"{len(extra_metadata)}"
    )

    if missing_metadata or extra_metadata:
        raise RuntimeError(
            "Ground truth and metadata image IDs do not match."
        )

    print(
        "PASS  Ground truth ↔ metadata correspondence"
    )

    # =========================================================
    # 6. Physical image files
    # =========================================================

    print(
        "\n[4] Physical Image Files"
    )

    image_files = list(
        IMAGE_DIR.glob("*.jpg")
    )

    print(
        f"JPEG files found: {len(image_files)}"
    )

    if len(image_files) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} JPEG files, "
            f"found {len(image_files)}"
        )

    physical_ids = {
        path.stem
        for path in image_files
    }

    missing_images = gt_ids - physical_ids
    extra_images = physical_ids - gt_ids

    print(
        f"Missing physical images: {len(missing_images)}"
    )

    print(
        f"Extra physical images:   {len(extra_images)}"
    )

    if missing_images:

        print(
            "First missing:"
        )

        print(
            sorted(missing_images)[:20]
        )

        raise RuntimeError(
            "Ground truth images missing on disk."
        )

    if extra_images:

        print(
            "First extra:"
        )

        print(
            sorted(extra_images)[:20]
        )

        raise RuntimeError(
            "Physical images not represented in metadata."
        )

    print(
        "PASS  Image ↔ metadata correspondence"
    )

    # =========================================================
    # 7. Downsampled statistics
    # =========================================================

    downsampled_ids = {
        image_id
        for image_id in physical_ids
        if image_id.endswith("_downsampled")
    }

    standard_ids = (
        physical_ids - downsampled_ids
    )

    print(
        "\n[5] Filename Statistics"
    )

    print(
        f"Downsampled images: {len(downsampled_ids)}"
    )

    print(
        f"Standard images:    {len(standard_ids)}"
    )

    if len(downsampled_ids) != 2074:
        raise RuntimeError(
            "Unexpected downsampled image count."
        )

    print(
        "PASS  Filename statistics"
    )

    # =========================================================
    # 8. Lesion ID statistics
    # =========================================================

    print(
        "\n[6] Lesion Identity"
    )

    unique_lesions = meta[
        "lesion_id"
    ].nunique(
        dropna=True
    )

    missing_lesions = int(
        meta["lesion_id"].isna().sum()
    )

    identified_images = int(
        meta["lesion_id"].notna().sum()
    )

    print(
        f"Unique lesion IDs:       {unique_lesions}"
    )

    print(
        f"Images with lesion ID:   {identified_images}"
    )

    print(
        f"Images without lesion ID:{missing_lesions}"
    )

    # =========================================================
    # 9. Verify image readability
    # =========================================================

    print(
        "\n[7] Image Decode Validation"
    )

    corrupted = []
    checked = 0

    for image_path in image_files:

        try:

            with Image.open(image_path) as img:

                img.verify()

            checked += 1

        except Exception as exc:

            corrupted.append(
                (
                    image_path.name,
                    str(exc)
                )
            )

    print(
        f"Images checked: {checked}"
    )

    print(
        f"Corrupted/unreadable: {len(corrupted)}"
    )

    if corrupted:

        print(
            "\nFirst corrupted images:"
        )

        for filename, error in corrupted[:20]:

            print(
                f"{filename}: {error}"
            )

        raise RuntimeError(
            "Corrupted or unreadable images detected."
        )

    print(
        "PASS  All JPEG images are readable"
    )

    # =========================================================
    # 10. Final summary
    # =========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "STATUS: PASS"
    )

    print(
        "ISIC 2019 raw dataset is internally consistent."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
