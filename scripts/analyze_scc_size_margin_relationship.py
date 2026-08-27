from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


C1_GEOMETRY = Path(
    "analysis/scc_bcc/"
    "c1_lesion_geometry/"
    "scc_lesion_geometry.csv"
)

F1_GEOMETRY = Path(
    "analysis/scc_bcc/"
    "f1/"
    "scc_lesion_geometry.csv"
)

METADATA = Path(
    "data/raw/pad_ufes/metadata.csv"
)

OUTPUT_DIR = Path(
    "analysis/scc_bcc/"
    "size_margin_relationship"
)


def normalize_lesion_id(value):
    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def load_metadata():
    df = pd.read_csv(METADATA)

    df["patient_id"] = (
        df["patient_id"].astype(str)
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
    ]

    df = df[columns]

    keys = [
        "patient_id",
        "lesion_id_key",
    ]

    # Metadata should be constant across images
    # belonging to the same lesion.
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
                    "Metadata differs across images "
                    f"for {patient_id}/{lesion_id}: "
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


def load_geometry(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} geometry missing: {path}"
        )

    df = pd.read_csv(path)

    # Older geometry files used the name
    # "scc_advantage_over_bcc". Normalize that legacy
    # name to the current convention.
    if (
        "bcc_minus_scc_distance" not in df.columns
        and "scc_advantage_over_bcc" in df.columns
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
        "error_status",
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
        df["patient_id"].astype(str)
    )

    df["lesion_id_key"] = (
        df["lesion_uid"]
        .astype(str)
        .str.rsplit("__", n=1)
        .str[-1]
    )

    df["geometry_label"] = label

    return df


def prepare_geometry(
    geometry,
    metadata,
):
    keys = [
        "patient_id",
        "lesion_id_key",
    ]

    merged = geometry.merge(
        metadata,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not (
        merged["_merge"] == "both"
    ).all():
        missing = merged[
            merged["_merge"] != "both"
        ]

        raise RuntimeError(
            "Missing metadata for:\n"
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

    merged["diameter_1"] = pd.to_numeric(
        merged["diameter_1"],
        errors="coerce",
    )

    merged["diameter_2"] = pd.to_numeric(
        merged["diameter_2"],
        errors="coerce",
    )

    merged["lesion_area_proxy"] = (
        merged["diameter_1"]
        * merged["diameter_2"]
    )

    boolean_columns = [
        "smoke",
        "drink",
        "pesticide",
        "skin_cancer_history",
        "cancer_history",
        "has_piped_water",
        "has_sewage_system",
        "itch",
        "hurt",
        "bleed",
        "elevation",
    ]

    for column in boolean_columns:
        if column in merged.columns:
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

    merged["problematic"] = (
        merged["error_status"]
        == "SCC_to_BCC_error"
    )

    return merged


def correlation_analysis(
    df,
    x_column,
    y_column,
):
    valid = df[
        [x_column, y_column]
    ].dropna()

    if len(valid) < 4:
        return {
            "n": len(valid),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    x = valid[x_column].to_numpy(
        dtype=float
    )

    y = valid[y_column].to_numpy(
        dtype=float
    )

    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)

    return {
        "n": len(valid),
        "pearson_r": float(
            pearson.statistic
        ),
        "pearson_p": float(
            pearson.pvalue
        ),
        "spearman_rho": float(
            spearman.statistic
        ),
        "spearman_p": float(
            spearman.pvalue
        ),
    }


def leave_one_out_spearman(
    df,
    x_column,
    y_column,
):
    valid = df[
        [x_column, y_column]
    ].dropna()

    if len(valid) < 5:
        return {
            "min_rho": np.nan,
            "max_rho": np.nan,
            "median_rho": np.nan,
            "negative_count": np.nan,
            "positive_count": np.nan,
            "n_runs": 0,
        }

    rhos = []

    values = valid.reset_index(
        drop=True
    )

    for index in range(len(values)):
        subset = values.drop(
            index
        )

        result = spearmanr(
            subset[x_column],
            subset[y_column],
        )

        rhos.append(
            float(result.statistic)
        )

    return {
        "min_rho": float(np.min(rhos)),
        "max_rho": float(np.max(rhos)),
        "median_rho": float(
            np.median(rhos)
        ),
        "negative_count": int(
            sum(rho < 0 for rho in rhos)
        ),
        "positive_count": int(
            sum(rho > 0 for rho in rhos)
        ),
        "n_runs": len(rhos),
    }


def print_correlations(
    label,
    df,
):
    print()
    print("=" * 80)
    print(f"{label}: SIZE ↔ FEATURE MARGIN")
    print("=" * 80)

    rows = []

    for size_column in (
        "diameter_1",
        "diameter_2",
        "lesion_area_proxy",
    ):
        result = correlation_analysis(
            df,
            size_column,
            "bcc_minus_scc_distance",
        )

        loo = leave_one_out_spearman(
            df,
            size_column,
            "bcc_minus_scc_distance",
        )

        print()
        print(size_column)
        print("-" * 70)

        print(
            f"N:              {result['n']}"
        )

        print(
            f"Pearson r:      "
            f"{result['pearson_r']:.6f}"
        )

        print(
            f"Pearson p:      "
            f"{result['pearson_p']:.6f}"
        )

        print(
            f"Spearman rho:   "
            f"{result['spearman_rho']:.6f}"
        )

        print(
            f"Spearman p:     "
            f"{result['spearman_p']:.6f}"
        )

        print(
            "LOO Spearman:   "
            f"min={loo['min_rho']:.6f}, "
            f"max={loo['max_rho']:.6f}, "
            f"median={loo['median_rho']:.6f}"
        )

        print(
            "LOO sign:       "
            f"{loo['positive_count']}/"
            f"{loo['n_runs']} positive, "
            f"{loo['negative_count']}/"
            f"{loo['n_runs']} negative"
        )

        rows.append(
            {
                "geometry": label,
                "size_variable": size_column,
                **result,
                **{
                    f"loo_{key}": value
                    for key, value in loo.items()
                },
            }
        )

    return rows


def print_group_summary(df, label):
    print()
    print("=" * 80)
    print(f"{label}: SIZE BY ERROR STATUS")
    print("=" * 80)

    for column in (
        "diameter_1",
        "diameter_2",
        "lesion_area_proxy",
    ):
        print()
        print(column)

        for group_name in (
            False,
            True,
        ):
            subset = df[
                df["problematic"]
                == group_name
            ][column].dropna()

            name = (
                "problematic"
                if group_name
                else "clean"
            )

            print(
                f"  {name:12s}: "
                f"n={len(subset):2d} "
                f"mean={subset.mean():.4f} "
                f"median={subset.median():.4f}"
            )


def print_hurt_relationship(df, label):
    print()
    print("=" * 80)
    print(f"{label}: HURT ↔ LESION SIZE")
    print("=" * 80)

    for column in (
        "diameter_1",
        "diameter_2",
        "lesion_area_proxy",
    ):
        subset = df[
            [column, "hurt"]
        ].copy()

        subset[column] = pd.to_numeric(
            subset[column],
            errors="coerce",
        )

        subset = subset.dropna()

        print()
        print(column)

        for value in (
            False,
            True,
        ):
            values = subset[
                subset["hurt"]
                == value
            ][column]

            name = (
                "hurt=True"
                if value
                else "hurt=False"
            )

            print(
                f"  {name:12s}: "
                f"n={len(values):2d} "
                f"mean={values.mean():.4f} "
                f"median={values.median():.4f}"
            )

        if (
            subset["hurt"]
            .nunique()
            == 2
        ):
            result = mann_whitney_safe(
                subset,
                column,
                "hurt",
            )

            print(
                f"  Mann-Whitney p="
                f"{result:.6f}"
            )


def mann_whitney_safe(
    df,
    numeric_column,
    boolean_column,
):
    from scipy.stats import mannwhitneyu

    a = df[
        df[boolean_column] == True
    ][numeric_column].dropna()

    b = df[
        df[boolean_column] == False
    ][numeric_column].dropna()

    if len(a) == 0 or len(b) == 0:
        return np.nan

    return float(
        mannwhitneyu(
            a,
            b,
            alternative="two-sided",
        ).pvalue
    )


def main():
    print("=" * 80)
    print(
        "DERMASENSE SCC SIZE–MARGIN "
        "RELATIONSHIP ANALYSIS"
    )
    print("=" * 80)

    metadata = load_metadata()

    c1 = prepare_geometry(
        load_geometry(
            C1_GEOMETRY,
            "C1",
        ),
        metadata,
    )

    f1 = prepare_geometry(
        load_geometry(
            F1_GEOMETRY,
            "F1",
        ),
        metadata,
    )

    if len(c1) != 22:
        raise RuntimeError(
            f"C1 expected 22 lesions, "
            f"found {len(c1)}"
        )

    if len(f1) != 22:
        raise RuntimeError(
            f"F1 expected 22 lesions, "
            f"found {len(f1)}"
        )

    c1_keys = set(
        c1["lesion_uid"]
    )

    f1_keys = set(
        f1["lesion_uid"]
    )

    if c1_keys != f1_keys:
        raise RuntimeError(
            "C1/F1 lesion identity mismatch."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Matched lesions: {len(c1)}"
    )

    c1_rows = print_correlations(
        "C1",
        c1,
    )

    f1_rows = print_correlations(
        "F1",
        f1,
    )

    print_group_summary(
        c1,
        "C1",
    )

    print_group_summary(
        f1,
        "F1",
    )

    print_hurt_relationship(
        c1,
        "C1",
    )

    print_hurt_relationship(
        f1,
        "F1",
    )

    correlation_df = pd.DataFrame(
        c1_rows + f1_rows
    )

    correlation_path = (
        OUTPUT_DIR
        / "size_margin_correlations.csv"
    )

    correlation_df.to_csv(
        correlation_path,
        index=False,
    )

    c1.to_csv(
        OUTPUT_DIR
        / "c1_scc_size_margin_data.csv",
        index=False,
    )

    f1.to_csv(
        OUTPUT_DIR
        / "f1_scc_size_margin_data.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        f"Correlations: {correlation_path}"
    )

    print(
        "C1 data: "
        f"{OUTPUT_DIR / 'c1_scc_size_margin_data.csv'}"
    )

    print(
        "F1 data: "
        f"{OUTPUT_DIR / 'f1_scc_size_margin_data.csv'}"
    )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
