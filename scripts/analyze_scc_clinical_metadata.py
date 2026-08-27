from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu


METADATA_PATH = Path(
    "data/raw/pad_ufes/metadata.csv"
)

GEOMETRY_PATH = Path(
    "analysis/scc_bcc/"
    "c1_lesion_geometry/"
    "scc_lesion_geometry.csv"
)

OUTPUT_DIR = Path(
    "analysis/scc_bcc/"
    "clinical_metadata"
)

NUMERIC_COLUMNS = (
    "age",
    "diameter_1",
    "diameter_2",
    "fitspatrick",
)

CATEGORICAL_COLUMNS = (
    "region",
    "gender",
    "smoke",
    "drink",
    "pesticide",
    "skin_cancer_history",
    "cancer_history",
    "itch",
    "grew",
    "hurt",
    "changed",
    "bleed",
    "elevation",
)


def normalize_lesion_id(value):
    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def cramers_v(table):
    """
    Bias-corrected Cramer's V for descriptive effect size.
    """
    observed = table.to_numpy(dtype=float)

    n = observed.sum()

    if n == 0:
        return float("nan")

    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)

    expected = np.outer(
        row_totals,
        col_totals,
    ) / n

    valid = expected > 0

    chi2 = np.sum(
        (
            (observed - expected) ** 2
            / np.where(valid, expected, 1)
        )[valid]
    )

    phi2 = chi2 / n
    r, k = observed.shape

    if n <= 1:
        return float("nan")

    phi2_corrected = max(
        0.0,
        phi2
        - ((k - 1) * (r - 1))
        / (n - 1),
    )

    r_corrected = (
        r
        - ((r - 1) ** 2)
        / (n - 1)
    )

    k_corrected = (
        k
        - ((k - 1) ** 2)
        / (n - 1)
    )

    denominator = min(
        k_corrected - 1,
        r_corrected - 1,
    )

    if denominator <= 0:
        return float("nan")

    return float(
        np.sqrt(
            phi2_corrected
            / denominator
        )
    )


def main():
    print("=" * 80)
    print("DERMASENSE SCC CLINICAL-METADATA AUDIT")
    print("=" * 80)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing metadata: {METADATA_PATH}"
        )

    if not GEOMETRY_PATH.exists():
        raise FileNotFoundError(
            f"Missing geometry: {GEOMETRY_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    geometry = pd.read_csv(
        GEOMETRY_PATH
    )

    required_geometry = {
        "patient_id",
        "lesion_uid",
        "error_status",
        "bcc_minus_scc_distance",
    }

    missing = (
        required_geometry
        - set(geometry.columns)
    )

    if missing:
        raise RuntimeError(
            "Geometry CSV is missing columns: "
            f"{sorted(missing)}"
        )

    # ------------------------------------------------------------
    # Restrict geometry to the 22 SCC lesions.
    # ------------------------------------------------------------

    geometry = geometry.copy()

    geometry["patient_id"] = (
        geometry["patient_id"]
        .astype(str)
    )

    # lesion_uid has the form PAT_492__937.
    geometry["lesion_id_key"] = (
        geometry["lesion_uid"]
        .astype(str)
        .str.rsplit("__", n=1)
        .str[-1]
    )

    geometry["group"] = np.where(
        geometry["error_status"]
        == "SCC_to_BCC_error",
        "problematic",
        "clean",
    )

    if len(geometry) != 22:
        raise RuntimeError(
            "Expected 22 SCC lesions in geometry, "
            f"found {len(geometry)}."
        )

    if geometry["lesion_uid"].duplicated().any():
        raise RuntimeError(
            "Geometry contains duplicate lesion_uid values."
        )

    # ------------------------------------------------------------
    # Normalize metadata lesion identity.
    # ------------------------------------------------------------

    metadata = metadata.copy()

    metadata["patient_id"] = (
        metadata["patient_id"]
        .astype(str)
    )

    metadata["lesion_id_key"] = (
        metadata["lesion_id"]
        .apply(normalize_lesion_id)
    )

    # Keep only SCC metadata.
    metadata = metadata[
        metadata["diagnostic"]
        .astype(str)
        .str.upper()
        == "SCC"
    ].copy()

    # ------------------------------------------------------------
    # Collapse image-level PAD-UFES metadata to lesion level.
    #
    # Clinical metadata should normally be identical for every
    # photograph of the same lesion. Verify that assumption.
    # ------------------------------------------------------------

    audit_columns = (
        list(NUMERIC_COLUMNS)
        + list(CATEGORICAL_COLUMNS)
    )

    keys = [
        "patient_id",
        "lesion_id_key",
    ]

    inconsistencies = []

    for (
        patient_id,
        lesion_id_key,
    ), group in metadata.groupby(
        keys,
        dropna=False,
    ):
        for column in audit_columns:
            values = (
                group[column]
                .dropna()
                .astype(str)
                .unique()
            )

            if len(values) > 1:
                inconsistencies.append(
                    {
                        "patient_id": patient_id,
                        "lesion_id": lesion_id_key,
                        "column": column,
                        "values": ";".join(values),
                    }
                )

    if inconsistencies:
        inconsistent_df = pd.DataFrame(
            inconsistencies
        )

        raise RuntimeError(
            "Clinical metadata differs across images "
            "of the same lesion:\n"
            + inconsistent_df.to_string(
                index=False
            )
        )

    lesion_metadata = (
        metadata.groupby(
            keys,
            as_index=False,
            dropna=False,
        )
        .first()
    )

    # ------------------------------------------------------------
    # Merge the 22 analyzed SCC lesions with clinical metadata.
    # ------------------------------------------------------------

    merged = geometry.merge(
        lesion_metadata[
            keys
            + audit_columns
        ],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_metadata = merged[
        merged["_merge"] != "both"
    ]

    if not missing_metadata.empty:
        raise RuntimeError(
            "Could not match metadata for these lesions:\n"
            + missing_metadata[
                [
                    "patient_id",
                    "lesion_uid",
                ]
            ].to_string(
                index=False
            )
        )

    merged = merged.drop(
        columns="_merge"
    )

    problematic = merged[
        merged["group"]
        == "problematic"
    ]

    clean = merged[
        merged["group"]
        == "clean"
    ]

    if len(problematic) != 11:
        raise RuntimeError(
            "Expected 11 problematic lesions, "
            f"found {len(problematic)}."
        )

    if len(clean) != 11:
        raise RuntimeError(
            "Expected 11 clean lesions, "
            f"found {len(clean)}."
        )

    print()
    print("MATCHING")
    print(f"Analyzed SCC lesions: {len(merged)}")
    print(f"Problematic:          {len(problematic)}")
    print(f"Clean:                {len(clean)}")
    print("Clinical matching:    PASS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_DIR
        / "scc_lesion_clinical_metadata.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Numeric comparisons.
    # ------------------------------------------------------------

    numeric_rows = []

    print()
    print("=" * 80)
    print("NUMERIC VARIABLES")
    print("=" * 80)

    for column in NUMERIC_COLUMNS:
        p = pd.to_numeric(
            problematic[column],
            errors="coerce",
        ).dropna()

        c = pd.to_numeric(
            clean[column],
            errors="coerce",
        ).dropna()

        print()
        print(column)
        print("-" * 60)

        print(
            f"Problematic: n={len(p)} "
            f"mean={p.mean():.3f} "
            f"median={p.median():.3f}"
        )

        print(
            f"Clean:       n={len(c)} "
            f"mean={c.mean():.3f} "
            f"median={c.median():.3f}"
        )

        if len(p) > 0 and len(c) > 0:
            result = mannwhitneyu(
                p,
                c,
                alternative="two-sided",
            )

            u = float(result.statistic)
            p_value = float(
                result.pvalue
            )

            # Rank-biserial correlation with sign:
            # positive => problematic values tend higher.
            rank_biserial = (
                (2.0 * u)
                / (len(p) * len(c))
                - 1.0
            )
        else:
            u = float("nan")
            p_value = float("nan")
            rank_biserial = float("nan")

        print(
            f"Mann-Whitney p={p_value:.6f}"
        )

        print(
            "Rank-biserial="
            f"{rank_biserial:.3f}"
        )

        numeric_rows.append(
            {
                "variable": column,
                "problematic_n": len(p),
                "clean_n": len(c),
                "problematic_mean": p.mean(),
                "clean_mean": c.mean(),
                "problematic_median": p.median(),
                "clean_median": c.median(),
                "mann_whitney_u": u,
                "p_value": p_value,
                "rank_biserial": rank_biserial,
            }
        )

    pd.DataFrame(
        numeric_rows
    ).to_csv(
        OUTPUT_DIR
        / "numeric_comparisons.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Categorical comparisons.
    # ------------------------------------------------------------

    categorical_rows = []

    print()
    print("=" * 80)
    print("CATEGORICAL VARIABLES")
    print("=" * 80)

    for column in CATEGORICAL_COLUMNS:
        table = pd.crosstab(
            merged["group"],
            merged[column],
            dropna=False,
        )

        print()
        print(column)
        print("-" * 60)
        print(table.to_string())

        effect = cramers_v(
            table
        )

        print(
            f"Cramer's V: {effect:.3f}"
        )

        fisher_p = float("nan")

        # Fisher's exact test only for true 2x2 tables.
        if table.shape == (2, 2):
            fisher_result = fisher_exact(
                table.to_numpy()
            )

            fisher_p = float(
                fisher_result.pvalue
            )

            print(
                f"Fisher exact p="
                f"{fisher_p:.6f}"
            )
        else:
            print(
                "Fisher exact p=N/A "
                "(not a 2x2 table)"
            )

        categorical_rows.append(
            {
                "variable": column,
                "levels": table.shape[1],
                "cramers_v": effect,
                "fisher_p_value": fisher_p,
            }
        )

    categorical_df = pd.DataFrame(
        categorical_rows
    ).sort_values(
        "cramers_v",
        ascending=False,
        na_position="last",
    )

    categorical_df.to_csv(
        OUTPUT_DIR
        / "categorical_effect_sizes.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Missingness.
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("MISSINGNESS")
    print("=" * 80)

    missing_rows = []

    for column in audit_columns:
        p_missing = int(
            problematic[column]
            .isna()
            .sum()
        )

        c_missing = int(
            clean[column]
            .isna()
            .sum()
        )

        missing_rows.append(
            {
                "variable": column,
                "problematic_missing": p_missing,
                "clean_missing": c_missing,
            }
        )

        if p_missing or c_missing:
            print(
                f"{column}: "
                f"problematic={p_missing}/11, "
                f"clean={c_missing}/11"
            )

    missing_df = pd.DataFrame(
        missing_rows
    )

    missing_df.to_csv(
        OUTPUT_DIR
        / "missingness.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Ranked descriptive effect sizes.
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("LARGEST DESCRIPTIVE CATEGORICAL EFFECTS")
    print("=" * 80)

    print(
        categorical_df[
            [
                "variable",
                "cramers_v",
                "fisher_p_value",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("IMPORTANT INTERPRETATION NOTE")
    print("=" * 80)

    print(
        "This is an exploratory confound audit with "
        "11 problematic and 11 clean SCC lesions."
    )

    print(
        "Multiple metadata variables are examined; "
        "individual uncorrected p-values are not "
        "confirmatory evidence."
    )

    print(
        "Prioritize effect size, missingness, and "
        "clinically interpretable clustering."
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        OUTPUT_DIR
        / "scc_lesion_clinical_metadata.csv"
    )

    print(
        OUTPUT_DIR
        / "numeric_comparisons.csv"
    )

    print(
        OUTPUT_DIR
        / "categorical_effect_sizes.csv"
    )

    print(
        OUTPUT_DIR
        / "missingness.csv"
    )

    print()
    print("=" * 80)
    print("CLINICAL-METADATA AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
