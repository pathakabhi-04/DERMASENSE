from pathlib import Path
import pandas as pd


ROOT = Path("data/raw/pad_ufes")
METADATA_PATH = ROOT / "metadata.csv"
IMAGE_ROOT = ROOT / "images"

OUTPUT = Path("data/manifests/pad_ufes_manifest.csv")


def main():

    print("=" * 60)
    print("BUILDING PAD-UFES-20 MANIFEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load metadata
    # ---------------------------------------------------------
    df = pd.read_csv(METADATA_PATH)

    print(f"\nLoaded metadata: {len(df)} rows")

    # ---------------------------------------------------------
    # Normalize image IDs
    # ---------------------------------------------------------
    df["image_id"] = df["img_id"].astype(str).map(
        lambda x: Path(x).name
    )

    df["patient_id"] = df["patient_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)

    # Canonical operational lesion identity.
    #
    # PAD-UFES lesion_id is not globally unique in the downloaded
    # metadata, so lesion identity must be scoped to patient.
    df["lesion_uid"] = (
        df["patient_id"]
        + "__"
        + df["lesion_id"]
    )

    # ---------------------------------------------------------
    # Resolve image paths
    # ---------------------------------------------------------
    image_lookup = {
        path.name: str(path)
        for path in IMAGE_ROOT.rglob("*.png")
    }

    df["image_path"] = df["image_id"].map(image_lookup)

    # ---------------------------------------------------------
    # Check that every image exists
    # ---------------------------------------------------------
    missing = df["image_path"].isna()

    if missing.any():

        print("\nERROR: Missing image files")

        print(
            df.loc[missing, "image_id"]
            .head(20)
            .to_string(index=False)
        )

        raise RuntimeError(
            f"{missing.sum()} metadata records have no image file."
        )

    # ---------------------------------------------------------
    # Dataset-level fields
    # ---------------------------------------------------------
    df["dataset"] = "pad_ufes"
    df["image_domain"] = "smartphone_clinical"

    # PAD-UFES native diagnosis is preserved.
    df["native_diagnosis"] = df["diagnostic"].astype(str)

    # ACK/NEV/SEK are clinical/non-biopsy-heavy categories;
    # BCC/SCC/MEL are biopsy-backed in the dataset's documented
    # diagnostic workflow. Keep this as a coarse provenance field,
    # not a substitute for the original biopsed column.
    df["label_strength"] = df["biopsed"].map(
        lambda x: "biopsy_backed" if str(x).lower() == "true"
        else "clinical"
    )

    # ---------------------------------------------------------
    # Select core manifest columns
    # ---------------------------------------------------------
    core_columns = [
        "dataset",
        "image_id",
        "patient_id",
        "lesion_id",
        "lesion_uid",
        "image_path",
        "native_diagnosis",
        "label_strength",
        "image_domain",
    ]

    manifest = df[core_columns].copy()

    # ---------------------------------------------------------
    # Sort deterministically
    # ---------------------------------------------------------
    manifest = manifest.sort_values(
        ["patient_id", "lesion_id", "image_id"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Final checks
    # ---------------------------------------------------------
    assert len(manifest) == 2298
    assert manifest["image_id"].nunique() == 2298

    # `lesion_id` is a source-native identifier and is not
    # globally unique in the downloaded metadata.
    #
    # `lesion_uid` is our operational lesion identity.
    assert manifest["lesion_uid"].nunique() == 1891

    assert manifest["patient_id"].nunique() == 1373

    assert manifest["image_path"].notna().all()
    assert manifest["native_diagnosis"].notna().all()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest.to_csv(
        OUTPUT,
        index=False
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print("\nManifest statistics")
    print("-" * 40)

    print(f"Images:          {len(manifest)}")
    print(
        f"Unique operational lesions: "
        f"{manifest['lesion_uid'].nunique()}"
    )

    print(
        f"Unique source lesion IDs: "
        f"{manifest['lesion_id'].nunique()}"
    )
    print(f"Unique patients: {manifest['patient_id'].nunique()}")

    print("\nNative diagnoses")
    print(
        manifest["native_diagnosis"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nLabel strength")
    print(
        manifest["label_strength"]
        .value_counts()
        .to_string()
    )

    print("\nImage domain")
    print(
        manifest["image_domain"]
        .value_counts()
        .to_string()
    )

    print(f"\nSaved manifest:")
    print(OUTPUT)

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
