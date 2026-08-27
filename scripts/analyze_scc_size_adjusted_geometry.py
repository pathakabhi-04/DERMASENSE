from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


C1_PATH = Path(
    "analysis/scc_bcc/"
    "size_margin_relationship/"
    "c1_scc_size_margin_data.csv"
)

F1_PATH = Path(
    "analysis/scc_bcc/"
    "size_margin_relationship/"
    "f1_scc_size_margin_data.csv"
)

METADATA_PATH = Path(
    "data/raw/pad_ufes/metadata.csv"
)

OUTPUT_DIR = Path(
    "analysis/scc_bcc/"
    "size_adjusted_geometry"
)


def normalize_lesion_id(value):
    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: "
            f"{METADATA_PATH}"
        )

    df = pd.read_csv(
        METADATA_PATH
    )

    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
    )

    df["lesion_id_key"] = (
        df["lesion_id"]
        .apply(normalize_lesion_id)
    )

    df = df[
        df["diagnostic"]
        .astype(str)
        .str.upper()
        == "SCC"
    ].copy()

    columns = [
        "patient_id",
        "lesion_id_key",
        "age",
        "diameter_1",
        "diameter_2",
        "gender",
        "fitspatrick",
        "region",
        "itch",
        "grew",
        "hurt",
        "changed",
        "bleed",
        "elevation",
        "skin_cancer_history",
        "cancer_history",
    ]

    missing = (
        set(columns)
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Clinical metadata missing columns: "
            f"{sorted(missing)}"
        )

    df = df[columns].copy()

    # Metadata should be constant across images
    # belonging to the same lesion.
    keys = [
        "patient_id",
        "lesion_id_key",
    ]

    for (
        patient_id,
        lesion_id,
    ), group in df.groupby(
        keys,
        dropna=False,
    ):
        for column in columns[2:]:
            values = (
                group[column]
                .dropna()
                .astype(str)
                .unique()
            )

            if len(values) > 1:
                raise RuntimeError(
                    "Clinical metadata differs across "
                    "images for "
                    f"{patient_id}/{lesion_id}: "
                    f"{column}={values}"
                )

    return (
        df.groupby(
            keys,
            as_index=False,
            dropna=False,
        )
        .first()
    )


def load_geometry(
    path: Path,
    label: str,
):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} geometry file not found: "
            f"{path}"
        )

    df = pd.read_csv(path)

    # Support both the current and legacy margin
    # column names.
    if (
        "bcc_minus_scc_distance"
        not in df.columns
        and "scc_advantage_over_bcc"
        in df.columns
    ):
        df = df.rename(
            columns={
                "scc_advantage_over_bcc":
                "bcc_minus_scc_distance"
            }
        )

    required = {
        "patient_id",
        "lesion_uid",
        "bcc_minus_scc_distance",
        "problematic",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{label} geometry missing columns: "
            f"{sorted(missing)}"
        )

    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
    )

    df["lesion_uid"] = (
        df["lesion_uid"]
        .astype(str)
    )

    # lesion_uid is of the form:
    # PAT_492__937
    df["lesion_id_key"] = (
        df["lesion_uid"]
        .str.rsplit(
            "__",
            n=1,
        )
        .str[-1]
        .apply(normalize_lesion_id)
    )

    df["bcc_minus_scc_distance"] = (
        pd.to_numeric(
            df["bcc_minus_scc_distance"],
            errors="coerce",
        )
    )

    df["problematic"] = (
        df["problematic"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    return df


def merge_clinical_metadata(
    geometry,
    metadata,
    label,
):
    keys = [
        "patient_id",
        "lesion_id_key",
    ]

    # Geometry files may already contain size columns
    # from the previous size-margin analysis. Clinical
    # metadata is the authoritative source for these
    # variables, so remove the duplicated geometry copies
    # before merging.
    clinical_columns = [
        "age",
        "diameter_1",
        "diameter_2",
        "gender",
        "fitspatrick",
        "region",
        "itch",
        "grew",
        "hurt",
        "changed",
        "bleed",
        "elevation",
        "skin_cancer_history",
        "cancer_history",
    ]

    geometry = geometry.drop(
        columns=[
            column
            for column in clinical_columns
            if column in geometry.columns
        ],
        errors="ignore",
    )

    metadata_subset = metadata[
        keys + clinical_columns
    ].copy()

    merged = geometry.merge(
        metadata_subset,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing = merged[
        merged["_merge"] != "both"
    ]

    if len(missing) > 0:
        raise RuntimeError(
            f"{label}: clinical metadata missing "
            "for lesions:\n"
            + missing[
                [
                    "patient_id",
                    "lesion_uid",
                ]
            ].to_string(index=False)
        )

    merged = merged.drop(
        columns="_merge"
    )

    merged["diameter_1"] = (
        pd.to_numeric(
            merged["diameter_1"],
            errors="coerce",
        )
    )

    merged["diameter_2"] = (
        pd.to_numeric(
            merged["diameter_2"],
            errors="coerce",
        )
    )

    merged["lesion_area_proxy"] = (
        merged["diameter_1"]
        * merged["diameter_2"]
    )

    # Normalize clinical boolean fields. PAD-UFES
    # metadata may be loaded as strings rather than
    # Python booleans.
    boolean_columns = [
        "hurt",
        "itch",
        "grew",
        "changed",
        "bleed",
        "elevation",
        "skin_cancer_history",
        "cancer_history",
    ]

    for column in boolean_columns:
        merged[column] = (
            merged[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
        )

    return merged


def check_alignment(
    c1,
    f1,
):
    c1_ids = set(
        c1["lesion_uid"]
    )

    f1_ids = set(
        f1["lesion_uid"]
    )

    if c1_ids != f1_ids:
        raise RuntimeError(
            "C1/F1 lesion identity mismatch."
        )

    if len(c1) != 22:
        raise RuntimeError(
            f"Expected 22 C1 lesions, "
            f"found {len(c1)}"
        )

    if len(f1) != 22:
        raise RuntimeError(
            f"Expected 22 F1 lesions, "
            f"found {len(f1)}"
        )


def describe_group(
    df,
    label,
):
    problematic = df[
        df["problematic"] == True
    ]

    clean = df[
        df["problematic"] == False
    ]

    print()
    print("=" * 80)
    print(
        f"{label}: RAW GROUP COMPARISON"
    )
    print("=" * 80)

    for column in [
        "diameter_1",
        "diameter_2",
        "lesion_area_proxy",
        "bcc_minus_scc_distance",
    ]:
        p = problematic[
            column
        ].dropna()

        c = clean[
            column
        ].dropna()

        test = mannwhitneyu(
            p,
            c,
            alternative="two-sided",
        )

        print()
        print(column)

        print(
            f"  problematic: "
            f"n={len(p)} "
            f"mean={p.mean():.4f} "
            f"median={p.median():.4f}"
        )

        print(
            f"  clean:       "
            f"n={len(c)} "
            f"mean={c.mean():.4f} "
            f"median={c.median():.4f}"
        )

        print(
            f"  Mann-Whitney p="
            f"{test.pvalue:.6f}"
        )


def size_stratified_analysis(
    df,
    label,
):
    print()
    print("=" * 80)
    print(
        f"{label}: SIZE-STRATIFIED GEOMETRY"
    )
    print("=" * 80)

    rows = []

    for size_column in [
        "diameter_1",
        "diameter_2",
    ]:
        temp = df.copy()

        threshold = temp[
            size_column
        ].median()

        temp["size_group"] = np.where(
            temp[size_column]
            <= threshold,
            "small_or_equal",
            "large",
        )

        print()
        print(
            f"SIZE VARIABLE: "
            f"{size_column}"
        )

        print(
            f"Median threshold: "
            f"{threshold:.4f}"
        )

        for size_group in [
            "small_or_equal",
            "large",
        ]:
            subset = temp[
                temp["size_group"]
                == size_group
            ]

            problematic = subset[
                subset["problematic"]
                == True
            ][
                "bcc_minus_scc_distance"
            ].dropna()

            clean = subset[
                subset["problematic"]
                == False
            ][
                "bcc_minus_scc_distance"
            ].dropna()

            print()
            print(
                f"  {size_group}"
            )

            print(
                f"    total n="
                f"{len(subset)}"
            )

            print(
                f"    problematic n="
                f"{len(problematic)}"
            )

            print(
                f"    clean n="
                f"{len(clean)}"
            )

            if (
                len(problematic) >= 2
                and len(clean) >= 2
            ):
                test = mannwhitneyu(
                    problematic,
                    clean,
                    alternative="two-sided",
                )

                print(
                    f"    problematic margin="
                    f"{problematic.mean():.6f}"
                )

                print(
                    f"    clean margin="
                    f"{clean.mean():.6f}"
                )

                print(
                    f"    p="
                    f"{test.pvalue:.6f}"
                )

                rows.append(
                    {
                        "geometry": label,
                        "size_variable":
                            size_column,
                        "size_group":
                            size_group,
                        "threshold":
                            threshold,
                        "n_total":
                            len(subset),
                        "n_problematic":
                            len(problematic),
                        "n_clean":
                            len(clean),
                        "problematic_mean_margin":
                            problematic.mean(),
                        "clean_mean_margin":
                            clean.mean(),
                        "mann_whitney_p":
                            test.pvalue,
                    }
                )

    return rows


def residualize_against_size(
    df,
    label,
):
    print()
    print("=" * 80)
    print(
        f"{label}: SIZE-ADJUSTED MARGIN "
        "RESIDUAL ANALYSIS"
    )
    print("=" * 80)

    rows = []

    for size_column in [
        "diameter_1",
        "diameter_2",
        "lesion_area_proxy",
    ]:
        subset = df[
            [
                size_column,
                "bcc_minus_scc_distance",
                "problematic",
            ]
        ].dropna().copy()

        x = subset[
            size_column
        ].to_numpy(dtype=float)

        y = subset[
            "bcc_minus_scc_distance"
        ].to_numpy(dtype=float)

        if len(subset) < 4:
            continue

        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )

        predicted = (
            intercept
            + slope * x
        )

        subset[
            "size_adjusted_margin"
        ] = y - predicted

        problematic = subset[
            subset["problematic"]
            == True
        ][
            "size_adjusted_margin"
        ]

        clean = subset[
            subset["problematic"]
            == False
        ][
            "size_adjusted_margin"
        ]

        test = mannwhitneyu(
            problematic,
            clean,
            alternative="two-sided",
        )

        rho, rho_p = spearmanr(
            x,
            y,
        )

        print()
        print(size_column)
        print("-" * 70)

        print(
            f"Size→margin Spearman rho="
            f"{rho:.6f}"
        )

        print(
            f"Size→margin p="
            f"{rho_p:.6f}"
        )

        print(
            f"Linear slope="
            f"{slope:.6f}"
        )

        print(
            f"Problematic adjusted margin="
            f"{problematic.mean():.6f}"
        )

        print(
            f"Clean adjusted margin="
            f"{clean.mean():.6f}"
        )

        print(
            f"Adjusted margin difference="
            f"{problematic.mean() - clean.mean():.6f}"
        )

        print(
            f"Mann-Whitney p="
            f"{test.pvalue:.6f}"
        )

        rows.append(
            {
                "geometry": label,
                "size_variable":
                    size_column,
                "size_margin_spearman_rho":
                    rho,
                "size_margin_spearman_p":
                    rho_p,
                "linear_slope":
                    slope,
                "problematic_adjusted_margin":
                    problematic.mean(),
                "clean_adjusted_margin":
                    clean.mean(),
                "adjusted_margin_difference":
                    problematic.mean()
                    - clean.mean(),
                "mann_whitney_p":
                    test.pvalue,
            }
        )

    return rows


def hurt_analysis(
    df,
    label,
):
    print()
    print("=" * 80)
    print(
        f"{label}: HURT AFTER SIZE STRATIFICATION"
    )
    print("=" * 80)

    rows = []

    for size_column in [
        "diameter_1",
        "diameter_2",
        "lesion_area_proxy",
    ]:
        temp = df.copy()

        threshold = temp[
            size_column
        ].median()

        temp["size_group"] = np.where(
            temp[size_column]
            <= threshold,
            "small_or_equal",
            "large",
        )

        print()
        print(size_column)

        for size_group in [
            "small_or_equal",
            "large",
        ]:
            subset = temp[
                temp["size_group"]
                == size_group
            ]

            hurt_true = subset[
                subset["hurt"] == True
            ]

            hurt_false = subset[
                subset["hurt"] == False
            ]

            print()
            print(
                f"  {size_group}:"
            )

            print(
                f"    hurt=True  n="
                f"{len(hurt_true)}"
            )

            print(
                f"    hurt=False n="
                f"{len(hurt_false)}"
            )

            if (
                len(hurt_true) > 0
                and len(hurt_false) > 0
            ):
                print(
                    f"    problematic fraction "
                    f"(hurt=True)="
                    f"{hurt_true['problematic'].mean():.3f}"
                )

                print(
                    f"    problematic fraction "
                    f"(hurt=False)="
                    f"{hurt_false['problematic'].mean():.3f}"
                )

                rows.append(
                    {
                        "geometry": label,
                        "size_variable":
                            size_column,
                        "size_group":
                            size_group,
                        "hurt_true_n":
                            len(hurt_true),
                        "hurt_false_n":
                            len(hurt_false),
                        "problematic_fraction_hurt":
                            hurt_true[
                                "problematic"
                            ].mean(),
                        "problematic_fraction_no_hurt":
                            hurt_false[
                                "problematic"
                            ].mean(),
                    }
                )

    return rows


def main():
    print("=" * 80)
    print(
        "DERMASENSE SCC SIZE-ADJUSTED "
        "CLINICAL/GEOMETRY AUDIT"
    )
    print("=" * 80)

    metadata = load_metadata()

    c1_geometry = load_geometry(
        C1_PATH,
        "C1",
    )

    f1_geometry = load_geometry(
        F1_PATH,
        "F1",
    )

    c1 = merge_clinical_metadata(
        c1_geometry,
        metadata,
        "C1",
    )

    f1 = merge_clinical_metadata(
        f1_geometry,
        metadata,
        "F1",
    )

    check_alignment(
        c1,
        f1,
    )

    print()
    print(
        f"Matched lesions: {len(c1)}"
    )

    print(
        "Clinical metadata matching: PASS"
    )

    all_results = []

    for label, df in [
        ("C1", c1),
        ("F1", f1),
    ]:
        describe_group(
            df,
            label,
        )

        all_results.extend(
            size_stratified_analysis(
                df,
                label,
            )
        )

        all_results.extend(
            residualize_against_size(
                df,
                label,
            )
        )

        hurt_analysis(
            df,
            label,
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        all_results
    ).to_csv(
        OUTPUT_DIR
        / "size_adjusted_geometry_results.csv",
        index=False,
    )

    c1.to_csv(
        OUTPUT_DIR
        / "c1_data.csv",
        index=False,
    )

    f1.to_csv(
        OUTPUT_DIR
        / "f1_data.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        "Results: "
        f"{OUTPUT_DIR / 'size_adjusted_geometry_results.csv'}"
    )

    print(
        "C1 data: "
        f"{OUTPUT_DIR / 'c1_data.csv'}"
    )

    print(
        "F1 data: "
        f"{OUTPUT_DIR / 'f1_data.csv'}"
    )

    print()
    print("=" * 80)
    print(
        "SIZE-ADJUSTED AUDIT COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()