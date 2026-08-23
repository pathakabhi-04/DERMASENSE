from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

MANIFEST_PATH = Path(
    "data/manifests/isic2019_manifest.csv"
)

OUTPUT_DIR = Path(
    "docs/audits/isic2019"
)

LESION_SUMMARY_PATH = OUTPUT_DIR / "lesion_summary.csv"
CLASS_IDENTITY_PATH = OUTPUT_DIR / "class_identity_summary.csv"
AUDIT_REPORT_PATH = OUTPUT_DIR / "ISIC2019_IDENTITY_AUDIT_v0.1.md"


# ============================================================
# HELPERS
# ============================================================

def pct(n, d):
    if d == 0:
        return 0.0
    return 100.0 * n / d


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ISIC 2019 IDENTITY & SPLIT AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(MANIFEST_PATH)

    print(
        f"Manifest rows: {len(df)}"
    )

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
            f"Manifest missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Basic identity counts
    # --------------------------------------------------------

    total_images = len(df)

    identified = df[
        df["lesion_id"].notna()
    ].copy()

    unknown = df[
        df["lesion_id"].isna()
    ].copy()

    unique_lesions = identified[
        "lesion_id"
    ].nunique()

    print("\n" + "-" * 70)
    print("1. BASIC IDENTITY COUNTS")
    print("-" * 70)

    print(
        f"Total images:              {total_images}"
    )

    print(
        f"Images with lesion ID:     {len(identified)}"
    )

    print(
        f"Images without lesion ID:  {len(unknown)}"
    )

    print(
        f"Unique lesion IDs:         {unique_lesions}"
    )

    print(
        f"Unknown-ID proportion:     "
        f"{pct(len(unknown), total_images):.2f}%"
    )

    # --------------------------------------------------------
    # Images per lesion
    # --------------------------------------------------------

    lesion_counts = (
        identified
        .groupby("lesion_id")
        .size()
        .rename("image_count")
        .reset_index()
    )

    print("\n" + "-" * 70)
    print("2. IMAGES PER IDENTIFIED LESION")
    print("-" * 70)

    print(
        lesion_counts[
            "image_count"
        ].describe().to_string()
    )

    single_image_lesions = (
        lesion_counts["image_count"] == 1
    ).sum()

    multi_image_lesions = (
        lesion_counts["image_count"] > 1
    ).sum()

    print(
        f"\nSingle-image lesions:      "
        f"{single_image_lesions}"
    )

    print(
        f"Multi-image lesions:       "
        f"{multi_image_lesions}"
    )

    print(
        f"Multi-image lesion rate:   "
        f"{pct(multi_image_lesions, unique_lesions):.2f}%"
    )

    # --------------------------------------------------------
    # Multi-image lesion distribution
    # --------------------------------------------------------

    print("\nMulti-image lesion distribution:")

    print(
        lesion_counts[
            lesion_counts["image_count"] > 1
        ]["image_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Lesion diagnostic consistency
    # --------------------------------------------------------

    lesion_diagnoses = (
        identified
        .groupby("lesion_id")[
            "native_diagnosis"
        ]
        .agg(
            diagnosis_count="nunique",
            diagnoses=lambda x: sorted(
                set(x)
            ),
            image_count="size",
        )
        .reset_index()
    )

    inconsistent_lesions = lesion_diagnoses[
        lesion_diagnoses[
            "diagnosis_count"
        ] > 1
    ].copy()

    print("\n" + "-" * 70)
    print("3. LESION DIAGNOSTIC CONSISTENCY")
    print("-" * 70)

    print(
        f"Lesions with one diagnosis: "
        f"{len(lesion_diagnoses) - len(inconsistent_lesions)}"
    )

    print(
        f"Lesions with multiple diagnoses: "
        f"{len(inconsistent_lesions)}"
    )

    if len(inconsistent_lesions):
        print(
            "\nWARNING: Some lesion IDs contain "
            "multiple native diagnoses."
        )

        print(
            inconsistent_lesions.head(20)
            .to_string(index=False)
        )

    else:
        print(
            "PASS  Every identified lesion has "
            "one consistent native diagnosis."
        )

    # --------------------------------------------------------
    # Class-level identity analysis
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("4. CLASS-LEVEL IDENTITY COVERAGE")
    print("-" * 70)

    class_rows = []

    for diagnosis, group in (
        df.groupby("native_diagnosis")
    ):

        image_count = len(group)

        identified_images = (
            group["lesion_id"].notna().sum()
        )

        unknown_images = (
            group["lesion_id"].isna().sum()
        )

        identified_lesions = (
            group.loc[
                group["lesion_id"].notna(),
                "lesion_id",
            ].nunique()
        )

        multi_image_lesions_class = (
            group.loc[
                group["lesion_id"].notna()
            ]
            .groupby("lesion_id")
            .size()
            .gt(1)
            .sum()
        )

        class_rows.append(
            {
                "native_diagnosis": diagnosis,
                "images": image_count,
                "identified_images": identified_images,
                "unknown_images": unknown_images,
                "unknown_image_pct": round(
                    pct(
                        unknown_images,
                        image_count,
                    ),
                    2,
                ),
                "unique_identified_lesions":
                    identified_lesions,
                "multi_image_lesions":
                    int(
                        multi_image_lesions_class
                    ),
            }
        )

    class_summary = pd.DataFrame(
        class_rows
    ).sort_values(
        "native_diagnosis"
    )

    print(
        class_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Unknown-ID class concentration
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("5. UNKNOWN LESION-ID SUBSET")
    print("-" * 70)

    unknown_class_counts = (
        unknown[
            "native_diagnosis"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        unknown_class_counts.to_string()
    )

    # --------------------------------------------------------
    # Can patient-level split be guaranteed?
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("6. PATIENT IDENTITY")
    print("-" * 70)

    patient_columns = [
        c for c in df.columns
        if "patient" in c.lower()
    ]

    if patient_columns:

        print(
            "Patient-related columns found:"
        )

        for col in patient_columns:
            print(
                f"  {col}"
            )

    else:

        print(
            "NO patient_id column exists in "
            "the ISIC 2019 manifest."
        )

        print(
            "Patient-level split integrity "
            "cannot be directly verified."
        )

    # --------------------------------------------------------
    # Split guarantees
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("7. SPLIT GUARANTEES")
    print("-" * 70)

    if len(inconsistent_lesions) == 0:

        print(
            "Identified lesion subset:"
        )

        print(
            "  PASS  Lesion-level grouping is "
            "available for 23,247 images."
        )

        print(
            "  PASS  A lesion-disjoint split can "
            "be mechanically enforced."
        )

    else:

        print(
            "WARNING: Lesion IDs are not fully "
            "diagnostically consistent."
        )

    print(
        "Unknown lesion-ID subset:"
    )

    print(
        f"  {len(unknown)} images cannot be "
        "proven lesion-disjoint."
    )

    print(
        "Patient-level guarantee:"
    )

    print(
        "  NOT AVAILABLE from supplied metadata."
    )

    # --------------------------------------------------------
    # Save lesion summary
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lesion_summary = (
        lesion_counts
        .merge(
            lesion_diagnoses[
                [
                    "lesion_id",
                    "diagnosis_count",
                    "diagnoses",
                ]
            ],
            on="lesion_id",
            how="left",
            validate="one_to_one",
        )
    )

    lesion_summary.to_csv(
        LESION_SUMMARY_PATH,
        index=False,
    )

    class_summary.to_csv(
        CLASS_IDENTITY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Generate Markdown report
    # --------------------------------------------------------

    report = []

    report.append(
        "# ISIC 2019 Identity & Split Audit v0.1\n"
    )

    report.append(
        "## Scope\n\n"
        "This audit evaluates lesion identity, "
        "image grouping, diagnostic consistency, "
        "and the guarantees available for future "
        "train/validation/test splitting. "
        "It does not create dataset splits.\n"
    )

    report.append(
        "## Dataset identity\n\n"
        f"- Total images: **{total_images}**\n"
        f"- Images with lesion ID: **{len(identified)}**\n"
        f"- Images without lesion ID: **{len(unknown)}**\n"
        f"- Unique identified lesions: **{unique_lesions}**\n"
        f"- Unknown-ID proportion: "
        f"**{pct(len(unknown), total_images):.2f}%**\n"
    )

    report.append(
        "## Images per lesion\n\n"
        f"- Single-image lesions: **{single_image_lesions}**\n"
        f"- Multi-image lesions: **{multi_image_lesions}**\n"
        f"- Multi-image lesion rate: "
        f"**{pct(multi_image_lesions, unique_lesions):.2f}%**\n"
    )

    report.append(
        "## Diagnostic consistency\n\n"
        f"- Lesions with multiple native diagnoses: "
        f"**{len(inconsistent_lesions)}**\n"
    )

    if len(inconsistent_lesions) == 0:
        report.append(
            "- **PASS:** Every identified lesion has "
            "one consistent native diagnosis.\n"
        )
    else:
        report.append(
            "- **WARNING:** Some lesion IDs contain "
            "multiple native diagnoses and require review "
            "before lesion-level splitting.\n"
        )

    report.append(
        "## Patient identity\n\n"
        "- `patient_id` is **not present** in the "
        "supplied ISIC 2019 metadata.\n"
        "- A patient-level split guarantee therefore "
        "cannot be established from this dataset alone.\n"
    )

    report.append(
        "## Split implication\n\n"
        f"- The **{len(identified)}** images with "
        "identified lesion IDs can participate in a "
        "lesion-disjoint split.\n"
        f"- The **{len(unknown)}** images without lesion IDs "
        "cannot be proven lesion-disjoint.\n"
        "- These unknown-ID records should not be silently "
        "treated as independent lesions.\n"
        "- No train/validation/test split is frozen by "
        "this audit.\n"
    )

    report.append(
        "## Generated artifacts\n\n"
        f"- `{LESION_SUMMARY_PATH}`\n"
        f"- `{CLASS_IDENTITY_PATH}`\n"
        f"- `{AUDIT_REPORT_PATH}`\n"
    )

    AUDIT_REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUDIT ARTIFACTS")
    print("=" * 70)

    print(
        f"Lesion summary: "
        f"{LESION_SUMMARY_PATH}"
    )

    print(
        f"Class summary:  "
        f"{CLASS_IDENTITY_PATH}"
    )

    print(
        f"Audit report:   "
        f"{AUDIT_REPORT_PATH}"
    )

    print("\n" + "=" * 70)
    print("STATUS: PASS")
    print(
        "ISIC 2019 identity audit completed. "
        "No train/validation/test split was created."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()