from pathlib import Path
import sys

import pandas as pd


MANIFEST_PATH = Path(
    "data/manifests/pad_ufes_manifest.csv"
)

SPLIT_DIR = Path(
    "data/splits/pad_ufes"
)

TRAIN_PATH = SPLIT_DIR / "train.csv"
VAL_PATH = SPLIT_DIR / "val.csv"
TEST_PATH = SPLIT_DIR / "test.csv"


EXPECTED_TOTAL_IMAGES = 2298
EXPECTED_TOTAL_PATIENTS = 1373
EXPECTED_SOURCE_LESION_IDS = 1641
EXPECTED_OPERATIONAL_LESIONS = 1891

EXPECTED_DIAGNOSES = {
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
}


REQUIRED_COLUMNS = {
    "dataset",
    "image_id",
    "patient_id",
    "lesion_id",
    "lesion_uid",
    "image_path",
    "native_diagnosis",
    "label_strength",
    "image_domain",
}


def fail(errors, message):
    errors.append(message)
    print(f"FAIL  {message}")


def check_file_exists(path, errors):
    if not path.exists():
        fail(errors, f"Missing file: {path}")
        return False

    return True


def check_columns(df, name, errors):
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        fail(
            errors,
            f"{name} missing columns: {sorted(missing)}"
        )


def check_duplicate_ids(df, name, errors):
    duplicate_images = df["image_id"].duplicated().sum()

    if duplicate_images:
        fail(
            errors,
            f"{name}: {duplicate_images} duplicate image_id values"
        )


def validate_split(
    df,
    split_name,
    manifest,
    errors,
):
    print(f"\n[{split_name.upper()}]")

    # ---------------------------------------------------------
    # Basic structure
    # ---------------------------------------------------------
    check_columns(df, split_name, errors)

    if "split" not in df.columns:
        fail(
            errors,
            f"{split_name}: missing 'split' column"
        )
    else:
        invalid = set(df["split"].dropna()) - {
            split_name
        }

        if invalid:
            fail(
                errors,
                f"{split_name}: unexpected split values {invalid}"
            )

    # ---------------------------------------------------------
    # Duplicate image IDs inside split
    # ---------------------------------------------------------
    duplicate_images = df["image_id"].duplicated().sum()

    if duplicate_images == 0:
        print("PASS  No duplicate image IDs")
    else:
        fail(
            errors,
            f"{split_name}: {duplicate_images} duplicate image IDs"
        )

    # ---------------------------------------------------------
    # Check images belong to original manifest
    # ---------------------------------------------------------
    manifest_images = set(
        manifest["image_id"]
    )

    split_images = set(
        df["image_id"]
    )

    unknown_images = split_images - manifest_images

    if not unknown_images:
        print("PASS  All image IDs exist in manifest")
    else:
        fail(
            errors,
            f"{split_name}: {len(unknown_images)} unknown image IDs"
        )

    # ---------------------------------------------------------
    # Check patient IDs belong to manifest
    # ---------------------------------------------------------
    manifest_patients = set(
        manifest["patient_id"]
    )

    split_patients = set(
        df["patient_id"]
    )

    unknown_patients = split_patients - manifest_patients

    if not unknown_patients:
        print("PASS  All patient IDs exist in manifest")
    else:
        fail(
            errors,
            f"{split_name}: {len(unknown_patients)} unknown patient IDs"
        )

    # ---------------------------------------------------------
    # Check lesion UIDs belong to manifest
    # ---------------------------------------------------------
    manifest_lesions = set(
        manifest["lesion_uid"]
    )

    split_lesions = set(
        df["lesion_uid"]
    )

    unknown_lesions = split_lesions - manifest_lesions

    if not unknown_lesions:
        print("PASS  All lesion UIDs exist in manifest")
    else:
        fail(
            errors,
            f"{split_name}: {len(unknown_lesions)} unknown lesion UIDs"
        )

    # ---------------------------------------------------------
    # Check diagnostic labels
    # ---------------------------------------------------------
    invalid_diagnoses = (
        set(df["native_diagnosis"])
        - EXPECTED_DIAGNOSES
    )

    if not invalid_diagnoses:
        print("PASS  Native diagnoses valid")
    else:
        fail(
            errors,
            f"{split_name}: invalid diagnoses {invalid_diagnoses}"
        )

    # ---------------------------------------------------------
    # Basic statistics
    # ---------------------------------------------------------
    print(
        f"Images:   {df['image_id'].nunique()}"
    )

    print(
        f"Lesions:  {df['lesion_uid'].nunique()}"
    )

    print(
        f"Patients: {df['patient_id'].nunique()}"
    )

    print("\nDiagnosis counts:")

    print(
        df["native_diagnosis"]
        .value_counts()
        .sort_index()
        .to_string()
    )


def main():

    print("=" * 70)
    print("PAD-UFES-20 INDEPENDENT SPLIT VALIDATOR")
    print("=" * 70)

    errors = []

    # ---------------------------------------------------------
    # Check required files
    # ---------------------------------------------------------
    required_files = [
        MANIFEST_PATH,
        TRAIN_PATH,
        VAL_PATH,
        TEST_PATH,
    ]

    for path in required_files:
        check_file_exists(path, errors)

    if errors:
        print("\nSTATUS: FAIL")
        sys.exit(1)

    # ---------------------------------------------------------
    # Load files
    # ---------------------------------------------------------
    manifest = pd.read_csv(MANIFEST_PATH)

    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)

    print("\nFiles loaded successfully.")

    print(
        f"Manifest images: {len(manifest)}"
    )

    print(
        f"Train rows:      {len(train)}"
    )

    print(
        f"Val rows:        {len(val)}"
    )

    print(
        f"Test rows:       {len(test)}"
    )

    # ---------------------------------------------------------
    # Manifest integrity
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("MANIFEST INTEGRITY")
    print("-" * 70)

    check_columns(
        manifest,
        "manifest",
        errors
    )

    duplicate_manifest_images = (
        manifest["image_id"]
        .duplicated()
        .sum()
    )

    if duplicate_manifest_images == 0:
        print("PASS  Manifest image IDs are unique")
    else:
        fail(
            errors,
            f"Manifest contains "
            f"{duplicate_manifest_images} duplicate image IDs"
        )

    if len(manifest) == EXPECTED_TOTAL_IMAGES:
        print(
            f"PASS  Manifest contains "
            f"{EXPECTED_TOTAL_IMAGES} images"
        )
    else:
        fail(
            errors,
            f"Expected {EXPECTED_TOTAL_IMAGES} manifest images, "
            f"found {len(manifest)}"
        )

    if (
        manifest["patient_id"].nunique()
        == EXPECTED_TOTAL_PATIENTS
    ):
        print(
            f"PASS  Manifest contains "
            f"{EXPECTED_TOTAL_PATIENTS} patients"
        )
    else:
        fail(
            errors,
            "Unexpected patient count: "
            f"{manifest['patient_id'].nunique()}"
        )

    if (
        manifest["lesion_id"].nunique()
        == EXPECTED_SOURCE_LESION_IDS
    ):
        print(
            f"PASS  Source lesion IDs: "
            f"{EXPECTED_SOURCE_LESION_IDS}"
        )
    else:
        fail(
            errors,
            "Unexpected source lesion-ID count: "
            f"{manifest['lesion_id'].nunique()}"
        )

    if (
        manifest["lesion_uid"].nunique()
        == EXPECTED_OPERATIONAL_LESIONS
    ):
        print(
            f"PASS  Operational lesions: "
            f"{EXPECTED_OPERATIONAL_LESIONS}"
        )
    else:
        fail(
            errors,
            "Unexpected operational lesion count: "
            f"{manifest['lesion_uid'].nunique()}"
        )

    # ---------------------------------------------------------
    # Individual split validation
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("INDIVIDUAL SPLIT VALIDATION")
    print("-" * 70)

    validate_split(
        train,
        "train",
        manifest,
        errors,
    )

    validate_split(
        val,
        "val",
        manifest,
        errors,
    )

    validate_split(
        test,
        "test",
        manifest,
        errors,
    )

    # ---------------------------------------------------------
    # Global image partition
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("GLOBAL PARTITION CHECK")
    print("-" * 70)

    manifest_images = set(
        manifest["image_id"]
    )

    train_images = set(train["image_id"])
    val_images = set(val["image_id"])
    test_images = set(test["image_id"])

    # ---------------------------------------------------------
    # Pairwise image overlap
    # ---------------------------------------------------------
    train_val = train_images & val_images
    train_test = train_images & test_images
    val_test = val_images & test_images

    if not train_val:
        print("PASS  Train ↔ Val image overlap: 0")
    else:
        fail(
            errors,
            f"Train ↔ Val image overlap: {len(train_val)}"
        )

    if not train_test:
        print("PASS  Train ↔ Test image overlap: 0")
    else:
        fail(
            errors,
            f"Train ↔ Test image overlap: {len(train_test)}"
        )

    if not val_test:
        print("PASS  Val ↔ Test image overlap: 0")
    else:
        fail(
            errors,
            f"Val ↔ Test image overlap: {len(val_test)}"
        )

    # ---------------------------------------------------------
    # Complete image coverage
    # ---------------------------------------------------------
    split_images = (
        train_images
        | val_images
        | test_images
    )

    missing_images = (
        manifest_images - split_images
    )

    extra_images = (
        split_images - manifest_images
    )

    if not missing_images:
        print("PASS  No manifest images missing from splits")
    else:
        fail(
            errors,
            f"{len(missing_images)} manifest images "
            f"are absent from splits"
        )

    if not extra_images:
        print("PASS  No unknown images in splits")
    else:
        fail(
            errors,
            f"{len(extra_images)} unknown images "
            f"found in splits"
        )

    if len(split_images) == EXPECTED_TOTAL_IMAGES:
        print(
            f"PASS  Exactly {EXPECTED_TOTAL_IMAGES} "
            f"unique images across splits"
        )
    else:
        fail(
            errors,
            "Unexpected total unique split images: "
            f"{len(split_images)}"
        )

    # ---------------------------------------------------------
    # Patient leakage
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("PATIENT LEAKAGE")
    print("-" * 70)

    train_patients = set(train["patient_id"])
    val_patients = set(val["patient_id"])
    test_patients = set(test["patient_id"])

    patient_overlaps = {
        "train_val": train_patients & val_patients,
        "train_test": train_patients & test_patients,
        "val_test": val_patients & test_patients,
    }

    for name, overlap in patient_overlaps.items():

        if not overlap:
            print(
                f"PASS  {name} patient overlap: 0"
            )
        else:
            fail(
                errors,
                f"{name} patient overlap: "
                f"{len(overlap)}"
            )

    # ---------------------------------------------------------
    # Operational lesion leakage
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("OPERATIONAL LESION LEAKAGE")
    print("-" * 70)

    train_lesions = set(train["lesion_uid"])
    val_lesions = set(val["lesion_uid"])
    test_lesions = set(test["lesion_uid"])

    lesion_overlaps = {
        "train_val": train_lesions & val_lesions,
        "train_test": train_lesions & test_lesions,
        "val_test": val_lesions & test_lesions,
    }

    for name, overlap in lesion_overlaps.items():

        if not overlap:
            print(
                f"PASS  {name} lesion overlap: 0"
            )
        else:
            fail(
                errors,
                f"{name} lesion overlap: "
                f"{len(overlap)}"
            )

    # ---------------------------------------------------------
    # Final class coverage
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("CLASS COVERAGE")
    print("-" * 70)

    for diagnosis in sorted(EXPECTED_DIAGNOSES):

        counts = {
            "train": int(
                (train["native_diagnosis"] == diagnosis).sum()
            ),
            "val": int(
                (val["native_diagnosis"] == diagnosis).sum()
            ),
            "test": int(
                (test["native_diagnosis"] == diagnosis).sum()
            ),
        }

        print(
            f"{diagnosis:5s} | "
            f"train={counts['train']:4d} | "
            f"val={counts['val']:3d} | "
            f"test={counts['test']:3d}"
        )

        for split_name, count in counts.items():

            if count == 0:
                fail(
                    errors,
                    f"{diagnosis} has zero images "
                    f"in {split_name}"
                )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    print("\n" + "=" * 70)

    if errors:

        print(
            f"STATUS: FAIL "
            f"({len(errors)} issue(s))"
        )

        print("\nIssues:")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")

        print("=" * 70)

        sys.exit(1)

    print("STATUS: PASS")
    print(
        "Independent validation confirms that the "
        "PAD-UFES-20 split artifacts are valid."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
