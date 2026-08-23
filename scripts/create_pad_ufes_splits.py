from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


MANIFEST = Path("data/manifests/pad_ufes_manifest.csv")
OUTPUT_DIR = Path("data/splits/pad_ufes")

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

DIAGNOSES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
]


def build_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert image-level manifest into one row per patient.

    Each diagnosis becomes a binary patient-level label:
        patient has ACK? 0/1
        patient has BCC? 0/1
        ...

    This allows multilabel stratification while keeping
    the patient as the split unit.
    """

    patient_diagnoses = (
        df.groupby("patient_id")["native_diagnosis"]
        .apply(lambda x: set(x))
        .reset_index(name="diagnoses")
    )

    for diagnosis in DIAGNOSES:
        patient_diagnoses[diagnosis] = patient_diagnoses[
            "diagnoses"
        ].apply(lambda diagnoses: int(diagnosis in diagnoses))

    return patient_diagnoses


def split_patients(patient_table: pd.DataFrame):
    """
    First split:
        70% train
        30% temporary

    Second split:
        temporary → 50% validation / 50% test

    Final:
        70 / 15 / 15
    """

    labels = patient_table[DIAGNOSES].values

    # ---------------------------------------------------------
    # Train / temporary
    # ---------------------------------------------------------
    first_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, temp_idx = next(
        first_splitter.split(patient_table, labels)
    )

    train_patients = patient_table.iloc[train_idx].copy()
    temp_patients = patient_table.iloc[temp_idx].copy()

    # ---------------------------------------------------------
    # Validation / test
    # ---------------------------------------------------------
    temp_labels = temp_patients[DIAGNOSES].values

    second_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )

    val_idx, test_idx = next(
        second_splitter.split(temp_patients, temp_labels)
    )

    val_patients = temp_patients.iloc[val_idx].copy()
    test_patients = temp_patients.iloc[test_idx].copy()

    return train_patients, val_patients, test_patients


def attach_split_to_manifest(
    manifest: pd.DataFrame,
    train_patients: pd.DataFrame,
    val_patients: pd.DataFrame,
    test_patients: pd.DataFrame,
):
    """
    Assign every image to the split belonging to its patient.
    """

    train_ids = set(train_patients["patient_id"])
    val_ids = set(val_patients["patient_id"])
    test_ids = set(test_patients["patient_id"])

    def assign_split(patient_id):
        if patient_id in train_ids:
            return "train"
        if patient_id in val_ids:
            return "val"
        if patient_id in test_ids:
            return "test"

        raise RuntimeError(
            f"Patient {patient_id} was not assigned to any split."
        )

    result = manifest.copy()

    result["split"] = result["patient_id"].map(assign_split)

    return result


def validate_splits(df: pd.DataFrame):
    """
    Hard leakage checks.
    """

    print("\n" + "=" * 60)
    print("SPLIT VALIDATION")
    print("=" * 60)

    errors = []

    # ---------------------------------------------------------
    # Split assignment
    # ---------------------------------------------------------
    if df["split"].isna().any():
        errors.append("Some images have no split assignment.")

    allowed = {"train", "val", "test"}

    invalid_splits = set(df["split"].dropna()) - allowed

    if invalid_splits:
        errors.append(
            f"Invalid split labels found: {invalid_splits}"
        )

    # ---------------------------------------------------------
    # Patient overlap
    # ---------------------------------------------------------
    train_patients = set(
        df.loc[df["split"] == "train", "patient_id"]
    )

    val_patients = set(
        df.loc[df["split"] == "val", "patient_id"]
    )

    test_patients = set(
        df.loc[df["split"] == "test", "patient_id"]
    )

    patient_overlap = (
        (train_patients & val_patients)
        | (train_patients & test_patients)
        | (val_patients & test_patients)
    )

    if patient_overlap:
        errors.append(
            f"Patient leakage detected: {len(patient_overlap)} patients."
        )

    # ---------------------------------------------------------
    # Lesion overlap
    # ---------------------------------------------------------
    train_lesions = set(
        df.loc[df["split"] == "train", "lesion_uid"]
    )

    val_lesions = set(
        df.loc[df["split"] == "val", "lesion_uid"]
    )

    test_lesions = set(
        df.loc[df["split"] == "test", "lesion_uid"]
    )

    lesion_overlap = (
        (train_lesions & val_lesions)
        | (train_lesions & test_lesions)
        | (val_lesions & test_lesions)
    )

    if lesion_overlap:
        errors.append(
            f"Lesion leakage detected: {len(lesion_overlap)} lesions."
        )

    # ---------------------------------------------------------
    # Image overlap
    # ---------------------------------------------------------
    train_images = set(
        df.loc[df["split"] == "train", "image_id"]
    )

    val_images = set(
        df.loc[df["split"] == "val", "image_id"]
    )

    test_images = set(
        df.loc[df["split"] == "test", "image_id"]
    )

    image_overlap = (
        (train_images & val_images)
        | (train_images & test_images)
        | (val_images & test_images)
    )

    if image_overlap:
        errors.append(
            f"Image leakage detected: {len(image_overlap)} images."
        )

    # ---------------------------------------------------------
    # Coverage
    # ---------------------------------------------------------
    split_counts = df["split"].value_counts()

    if set(split_counts.index) != allowed:
        errors.append(
            f"Expected all three splits, got: {set(split_counts.index)}"
        )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print(
        f"Train: {len(train_images):4d} images | "
        f"{len(train_lesions):4d} lesions | "
        f"{len(train_patients):4d} patients"
    )

    print(
        f"Val:   {len(val_images):4d} images | "
        f"{len(val_lesions):4d} lesions | "
        f"{len(val_patients):4d} patients"
    )

    print(
        f"Test:  {len(test_images):4d} images | "
        f"{len(test_lesions):4d} lesions | "
        f"{len(test_patients):4d} patients"
    )

    print("\nLeakage checks:")
    print(f"Patient overlap: {len(patient_overlap)}")
    print(f"Lesion overlap:  {len(lesion_overlap)}")
    print(f"Image overlap:   {len(image_overlap)}")

    if errors:
        print("\nSTATUS: FAIL")

        for error in errors:
            print(f"ERROR: {error}")

        raise RuntimeError(
            "Split validation failed."
        )

    print("\nSTATUS: PASS")
    print("Patient/lesion/image split integrity verified.")


def create_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an image-level split summary by native diagnosis.
    """

    summary = (
        df.groupby(["split", "native_diagnosis"])
        .agg(
            images=("image_id", "nunique"),
            lesions=("lesion_uid", "nunique"),
            patients=("patient_id", "nunique"),
        )
        .reset_index()
    )

    return summary


def main():

    print("=" * 60)
    print("PAD-UFES-20 PATIENT-LEVEL SPLIT GENERATOR")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load manifest
    # ---------------------------------------------------------
    manifest = pd.read_csv(MANIFEST)

    print(f"\nLoaded manifest: {len(manifest)} images")

    # ---------------------------------------------------------
    # Build patient-level labels
    # ---------------------------------------------------------
    patient_table = build_patient_table(manifest)

    print(
        f"Unique patients available: "
        f"{len(patient_table)}"
    )

    # ---------------------------------------------------------
    # Split patients
    # ---------------------------------------------------------
    train_patients, val_patients, test_patients = split_patients(
        patient_table
    )

    print("\nPatient allocation:")
    print(f"Train: {len(train_patients)}")
    print(f"Val:   {len(val_patients)}")
    print(f"Test:  {len(test_patients)}")

    # ---------------------------------------------------------
    # Attach split to every image
    # ---------------------------------------------------------
    split_manifest = attach_split_to_manifest(
        manifest,
        train_patients,
        val_patients,
        test_patients,
    )

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------
    validate_splits(split_manifest)

    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Save individual split manifests
    # ---------------------------------------------------------
    train_df = split_manifest[
        split_manifest["split"] == "train"
    ].copy()

    val_df = split_manifest[
        split_manifest["split"] == "val"
    ].copy()

    test_df = split_manifest[
        split_manifest["split"] == "test"
    ].copy()

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    val_df.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    summary = create_summary(split_manifest)

    summary.to_csv(
        OUTPUT_DIR / "split_summary.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("DIAGNOSIS DISTRIBUTION")
    print("=" * 60)

    print(
        summary.to_string(index=False)
    )

    print("\nFiles created:")

    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "val.csv")
    print(OUTPUT_DIR / "test.csv")
    print(OUTPUT_DIR / "split_summary.csv")

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
