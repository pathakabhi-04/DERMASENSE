from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/raw/itobos")

TRAIN_ROOT = DATA_ROOT / "_train" / "_train"
TRAIN_IMAGES = TRAIN_ROOT / "images"
TRAIN_LABELS = TRAIN_ROOT / "labels"
TRAIN_METADATA = TRAIN_ROOT / "metadata.csv"

TEST_ROOT = DATA_ROOT / "_test"
TEST_IMAGES = TEST_ROOT / "images"
TEST_METADATA = TEST_ROOT / "metadata.csv"

OUTPUT_DIR = Path("data/splits/itobos_detection")

SEED = 42
VAL_SIZE = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def lesion_count(label_path: Path) -> int:
    """Return the number of annotated lesion boxes in a YOLO label file."""
    if not label_path.exists():
        raise RuntimeError(f"Missing label file: {label_path}")

    with label_path.open() as f:
        return sum(1 for line in f if line.strip())


def density_bucket(count: int) -> str:
    """Convert lesion count into the locked CV-2 density buckets."""
    if count == 0:
        return "0"
    if count <= 3:
        return "1-3"
    if count <= 9:
        return "4-9"
    return "10+"


def age_bucket(value) -> str:
    """
    Convert age into the audit buckets.

    'Unknown' remains its own category.
    """
    if pd.isna(value) or str(value).strip().lower() == "unknown":
        return "Unknown"

    age = float(value)

    if age < 30:
        return "<30"
    if age < 40:
        return "30-39"
    if age < 50:
        return "40-49"
    if age < 60:
        return "50-59"
    if age < 70:
        return "60-69"
    return "70+"


def build_training_manifest() -> pd.DataFrame:
    """Build the manifest from the official iToBoS training images."""
    if not TRAIN_IMAGES.exists():
        raise RuntimeError(
            f"Training image directory does not exist: {TRAIN_IMAGES}"
        )

    if not TRAIN_LABELS.exists():
        raise RuntimeError(
            f"Training label directory does not exist: {TRAIN_LABELS}"
        )

    if not TRAIN_METADATA.exists():
        raise RuntimeError(
            f"Training metadata does not exist: {TRAIN_METADATA}"
        )

    metadata = pd.read_csv(TRAIN_METADATA)

    images = sorted(TRAIN_IMAGES.glob("*.png"))

    if len(images) != 8473:
        raise RuntimeError(
            f"Expected 8473 official training images, got {len(images)}"
        )

    rows = []

    for image_path in images:
        image_id = image_path.stem
        label_path = TRAIN_LABELS / f"{image_id}.txt"

        count = lesion_count(label_path)

        rows.append(
            {
                "dataset": "itobos_detection",
                "image_id": image_id,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "lesion_count": count,
                "density_bucket": density_bucket(count),
            }
        )

    manifest = pd.DataFrame(rows)

    manifest = manifest.merge(
        metadata,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if len(manifest) != 8473:
        raise RuntimeError(
            f"Expected 8473 manifest rows, got {len(manifest)}"
        )

    if manifest["image_id"].duplicated().any():
        raise RuntimeError("Duplicate image IDs detected.")

    if manifest["age_at_baseline"].isna().any():
        raise RuntimeError(
            "Unexpected NaN values found in age_at_baseline."
        )

    return manifest


def build_test_manifest() -> pd.DataFrame:
    """
    Build the untouched official iToBoS test manifest.

    The competition test set has no public detection labels.
    """
    if not TEST_IMAGES.exists():
        raise RuntimeError(
            f"Test image directory does not exist: {TEST_IMAGES}"
        )

    if not TEST_METADATA.exists():
        raise RuntimeError(
            f"Test metadata does not exist: {TEST_METADATA}"
        )

    metadata = pd.read_csv(TEST_METADATA)
    images = sorted(TEST_IMAGES.glob("*.png"))

    if len(images) != 8481:
        raise RuntimeError(
            f"Expected 8481 official test images, got {len(images)}"
        )

    rows = []

    for image_path in images:
        rows.append(
            {
                "dataset": "itobos_detection",
                "image_id": image_path.stem,
                "image_path": str(image_path),
            }
        )

    manifest = pd.DataFrame(rows)

    manifest = manifest.merge(
        metadata,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if len(manifest) != 8481:
        raise RuntimeError(
            f"Expected 8481 test rows, got {len(manifest)}"
        )

    if manifest["image_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate test image IDs detected."
        )

    return manifest


def distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return the distribution relevant to the CV-2 split audit."""
    total = len(df)

    density = (
        df["density_bucket"]
        .value_counts()
        .reindex(["0", "1-3", "4-9", "10+"], fill_value=0)
    )

    return pd.DataFrame(
        {
            "images": density,
            "percent": (density / total * 100).round(2),
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 90)
    print("DERMASENSE CV-2 DETECTION DATASET SPLIT")
    print("=" * 90)

    df = build_training_manifest()

    print()
    print("Official iToBoS training images:", len(df))
    print("Total annotated boxes:", int(df["lesion_count"].sum()))
    print(
        "Zero-lesion images:",
        int((df["lesion_count"] == 0).sum()),
    )

    # -----------------------------------------------------------------------
    # Combined stratification key
    #
    # Primary requirement:
    #   preserve lesion-density distribution.
    #
    # Secondary requirement:
    #   preserve age distribution as much as practical.
    #
    # Age and density are combined into one deterministic stratification
    # label. Rare combinations are collapsed to density-only strata so that
    # train_test_split does not receive singleton strata.
    # -----------------------------------------------------------------------

    df["age_bucket"] = df["age_at_baseline"].apply(age_bucket)

    combined = (
        df["density_bucket"]
        + "__"
        + df["age_bucket"]
    )

    counts = combined.value_counts()

    # A stratum must contain at least two samples for stratified splitting.
    df["stratify_key"] = combined.where(
        combined.map(counts) >= 2,
        df["density_bucket"],
    )

    # -----------------------------------------------------------------------
    # Internal train / validation split
    #
    # Official iToBoS test remains completely untouched.
    # -----------------------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=VAL_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=df["stratify_key"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()

    # Remove implementation-only column from exported manifests.
    export_columns = [
        "dataset",
        "image_id",
        "image_path",
        "label_path",
        "lesion_count",
        "density_bucket",
        "age_at_baseline",
        "body_part",
        "sun_damage_level",
        "pixel_spacing",
    ]

    train_df = (
        train_df[export_columns]
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    train_df.insert(0, "split", "train")

    val_df = (
        val_df[export_columns]
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    val_df.insert(0, "split", "val")

    test_df = build_test_manifest()

    test_columns = [
        "dataset",
        "image_id",
        "image_path",
        "age_at_baseline",
        "body_part",
        "sun_damage_level",
        "pixel_spacing",
    ]

    test_df = (
        test_df[test_columns]
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    test_df.insert(0, "split", "test")

    # -----------------------------------------------------------------------
    # Leakage checks
    # -----------------------------------------------------------------------

    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])
    test_ids = set(test_df["image_id"])

    # image_id is only unique within an iToBoS split.
    # The official train and test sets can reuse identifiers such as
    # image_0001 while referring to different image files.

    assert not train_ids & val_ids

    # Train/val are the only splits whose annotations participate in
    # our model-development split. The official test set is untouched
    # and therefore is not considered part of the development leakage check.

    assert len(train_ids) + len(val_ids) == 8473
    assert len(test_ids) == 8481

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    val_df.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    # -----------------------------------------------------------------------
    # Split summary
    # -----------------------------------------------------------------------

    summary_rows = []

    for name, split in [
        ("train", train_df),
        ("val", val_df),
    ]:
        density = (
            split["density_bucket"]
            .value_counts()
            .reindex(
                ["0", "1-3", "4-9", "10+"],
                fill_value=0,
            )
        )

        summary_rows.append(
            {
                "split": name,
                "images": len(split),
                "boxes": int(split["lesion_count"].sum()),
                "zero_lesion_images": int(
                    (split["lesion_count"] == 0).sum()
                ),
                "density_0": int(density["0"]),
                "density_1_3": int(density["1-3"]),
                "density_4_9": int(density["4-9"]),
                "density_10_plus": int(density["10+"]),
            }
        )

    # Official test contains no public ground-truth boxes, so do not invent
    # detection statistics for it.
    summary_rows.append(
        {
            "split": "test",
            "images": len(test_df),
            "boxes": "",
            "zero_lesion_images": "",
            "density_0": "",
            "density_1_3": "",
            "density_4_9": "",
            "density_10_plus": "",
        }
    )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "split_summary.csv",
        index=False,
    )

    # -----------------------------------------------------------------------
    # Human-readable audit
    # -----------------------------------------------------------------------

    print()
    print("-" * 90)
    print("SPLIT SUMMARY")
    print("-" * 90)
    print(summary_df.to_string(index=False))

    print()
    print("-" * 90)
    print("DENSITY DISTRIBUTION")
    print("-" * 90)

    for name, split in [
        ("TRAIN", train_df),
        ("VAL", val_df),
    ]:
        table = distribution_table(split)

        print()
        print(name)
        print(table.to_string())

    print()
    print("-" * 90)
    print("AGE DISTRIBUTION")
    print("-" * 90)

    for name, split in [
        ("TRAIN", train_df),
        ("VAL", val_df),
    ]:
        buckets = (
            split["age_at_baseline"]
            .apply(age_bucket)
            .value_counts()
            .reindex(
                [
                    "<30",
                    "30-39",
                    "40-49",
                    "50-59",
                    "60-69",
                    "70+",
                    "Unknown",
                ],
                fill_value=0,
            )
        )

        print()
        print(name)
        print(buckets.to_string())

    print()
    print("-" * 90)
    print("LEAKAGE CHECKS")
    print("-" * 90)
    print("Train ∩ Val: ", len(train_ids & val_ids))
    print()
    print("Train images:", len(train_ids))
    print("Val images:  ", len(val_ids))
    print("Official Test images:", len(test_ids))

    assert len(train_ids | val_ids) == 8473
    assert len(train_ids & val_ids) == 0

    print()
    print("=" * 90)
    print("CV-2 SPLIT CREATION COMPLETE")
    print("=" * 90)
    print()
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
