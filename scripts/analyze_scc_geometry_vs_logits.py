from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare C1→F1 SCC embedding displacement "
            "with classifier-logit displacement."
        )
    )

    parser.add_argument(
        "--geometry",
        default=(
            "analysis/scc_bcc/f1/"
            "c1_vs_f1_lesion_displacement.csv"
        ),
    )

    parser.add_argument(
        "--logits",
        default=(
            "analysis/scc_bcc/logit_analysis/"
            "scc_lesion_logits.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "analysis/scc_bcc/"
            "geometry_vs_logits"
        ),
    )

    return parser.parse_args()


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    result = pearsonr(x, y)

    return float(result.statistic), float(result.pvalue)


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    result = spearmanr(x, y)

    return float(result.statistic), float(result.pvalue)


def summarize_group(df, group_name):
    subset = df.copy()

    if group_name != "all":
        subset = subset[
            subset["group"] == group_name
        ].copy()

    n = len(subset)

    print()
    print("=" * 80)

    if group_name == "all":
        print("ALL SCC LESIONS")
    else:
        print(group_name.upper())

    print("=" * 80)

    print(f"N lesions: {n}")

    if n == 0:
        return {
            "group": group_name,
            "n": 0,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "same_direction_fraction": np.nan,
        }

    x = subset[
        "delta_embedding_margin"
    ].to_numpy(dtype=float)

    y = subset[
        "delta_logit_margin"
    ].to_numpy(dtype=float)

    pearson_r, pearson_p = safe_pearson(
        x,
        y,
    )

    spearman_rho, spearman_p = safe_spearman(
        x,
        y,
    )

    # A positive embedding delta means movement toward BCC
    # because the geometry margin is BCC-minus-SCC.
    #
    # A positive logit delta means movement toward SCC
    # because the classifier margin is SCC-minus-BCC.
    #
    # Therefore, "same semantic direction" requires
    # opposite signs between the two deltas.
    nonzero = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x != 0)
        & (y != 0)
    )

    if nonzero.any():
        same_direction = (
            np.sign(x[nonzero])
            != np.sign(y[nonzero])
        )

        same_direction_fraction = float(
            np.mean(same_direction)
        )
    else:
        same_direction_fraction = np.nan

    print()
    print("EMBEDDING → LOGIT ASSOCIATION")
    print("-" * 80)

    print(
        f"Pearson r:       "
        f"{pearson_r:.6f}"
    )

    print(
        f"Pearson p:       "
        f"{pearson_p:.6f}"
    )

    print(
        f"Spearman rho:    "
        f"{spearman_rho:.6f}"
    )

    print(
        f"Spearman p:      "
        f"{spearman_p:.6f}"
    )

    print()
    print(
        "Directional interpretation:"
    )

    print(
        "  Embedding Δ > 0 = movement toward BCC"
    )

    print(
        "  Embedding Δ < 0 = movement toward SCC"
    )

    print(
        "  Logit Δ > 0 = movement toward SCC"
    )

    print(
        "  Logit Δ < 0 = movement toward BCC"
    )

    print(
        f"Semantic-direction agreement: "
        f"{same_direction_fraction:.4f}"
    )

    return {
        "group": group_name,
        "n": n,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "same_direction_fraction": (
            same_direction_fraction
        ),
    }


def main():
    args = parse_args()

    geometry_path = Path(
        args.geometry
    )

    logits_path = Path(
        args.logits
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "DERMASENSE SCC GEOMETRY → LOGIT "
        "ALIGNMENT ANALYSIS"
    )
    print("=" * 80)

    print(
        f"Geometry: {geometry_path}"
    )

    print(
        f"Logits:   {logits_path}"
    )

    for path in (
        geometry_path,
        logits_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    geometry = pd.read_csv(
        geometry_path
    )

    logits = pd.read_csv(
        logits_path
    )

    required_geometry = {
        "patient_id",
        "lesion_uid",
        "group",
        "delta_margin",
    }

    required_logits = {
        "patient_id",
        "lesion_uid",
        "delta_logit_margin",
    }

    missing_geometry = (
        required_geometry
        - set(geometry.columns)
    )

    if missing_geometry:
        raise RuntimeError(
            "Geometry missing columns: "
            f"{sorted(missing_geometry)}"
        )

    missing_logits = (
        required_logits
        - set(logits.columns)
    )

    if missing_logits:
        raise RuntimeError(
            "Logit table missing columns: "
            f"{sorted(missing_logits)}"
        )

    geometry = geometry[
        [
            "patient_id",
            "lesion_uid",
            "group",
            "delta_margin",
        ]
    ].copy()

    logits = logits[
        [
            "patient_id",
            "lesion_uid",
            "delta_logit_margin",
        ]
    ].copy()

    geometry = geometry.rename(
        columns={
            "delta_margin":
                "delta_embedding_margin"
        }
    )

    # ------------------------------------------------------------
    # Validate lesion uniqueness.
    # ------------------------------------------------------------

    geometry_key_duplicates = (
        geometry[
            ["patient_id", "lesion_uid"]
        ]
        .duplicated()
        .any()
    )

    logits_key_duplicates = (
        logits[
            ["patient_id", "lesion_uid"]
        ]
        .duplicated()
        .any()
    )

    if geometry_key_duplicates:
        raise RuntimeError(
            "Duplicate lesion keys found "
            "in geometry table."
        )

    if logits_key_duplicates:
        raise RuntimeError(
            "Duplicate lesion keys found "
            "in logit table."
        )

    # ------------------------------------------------------------
    # Merge the exact same lesions.
    # ------------------------------------------------------------

    merged = geometry.merge(
        logits,
        on=[
            "patient_id",
            "lesion_uid",
        ],
        how="inner",
        validate="one_to_one",
    )

    print()
    print(
        f"Geometry lesions: {len(geometry)}"
    )

    print(
        f"Logit lesions:    {len(logits)}"
    )

    print(
        f"Matched lesions:  {len(merged)}"
    )

    if len(merged) != 22:
        raise RuntimeError(
            "Expected exactly 22 matched SCC lesions, "
            f"got {len(merged)}."
        )

    # ------------------------------------------------------------
    # Verify grouping.
    # ------------------------------------------------------------

    group_counts = (
        merged["group"]
        .value_counts()
        .to_dict()
    )

    print()
    print(
        "GROUP COUNTS"
    )
    print("-" * 80)

    for group in (
        "problematic",
        "clean",
    ):
        print(
            f"{group}: "
            f"{group_counts.get(group, 0)}"
        )

    if (
        group_counts.get("problematic", 0) != 11
        or group_counts.get("clean", 0) != 11
    ):
        raise RuntimeError(
            "Expected 11 problematic and "
            "11 clean lesions."
        )

    # ------------------------------------------------------------
    # Overall / subgroup statistics.
    # ------------------------------------------------------------

    summaries = []

    for group_name in (
        "all",
        "problematic",
        "clean",
    ):
        summaries.append(
            summarize_group(
                merged,
                group_name,
            )
        )

    summary_df = pd.DataFrame(
        summaries
    )

    # ------------------------------------------------------------
    # Print individual lesion displacement.
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "INDIVIDUAL LESION ALIGNMENT"
    )
    print("=" * 80)

    display_columns = [
        "patient_id",
        "lesion_uid",
        "group",
        "delta_embedding_margin",
        "delta_logit_margin",
    ]

    display_df = merged[
        display_columns
    ].sort_values(
        "delta_embedding_margin"
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------

    lesion_output = (
        output_dir
        / "geometry_vs_logits_lesions.csv"
    )

    summary_output = (
        output_dir
        / "geometry_vs_logits_summary.csv"
    )

    report_output = (
        output_dir
        / "geometry_vs_logits_summary.txt"
    )

    merged.to_csv(
        lesion_output,
        index=False,
    )

    summary_df.to_csv(
        summary_output,
        index=False,
    )

    with report_output.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "DERMASENSE SCC GEOMETRY → LOGIT "
            "ALIGNMENT ANALYSIS\n"
        )

        handle.write(
            "=" * 80 + "\n\n"
        )

        handle.write(
            f"Matched SCC lesions: "
            f"{len(merged)}\n"
        )

        handle.write(
            "Problematic: 11\n"
        )

        handle.write(
            "Clean:       11\n\n"
        )

        handle.write(
            "DIRECTION CONVENTIONS\n"
        )

        handle.write(
            "-" * 80 + "\n"
        )

        handle.write(
            "Embedding delta > 0 = movement toward BCC\n"
        )

        handle.write(
            "Embedding delta < 0 = movement toward SCC\n"
        )

        handle.write(
            "Logit delta > 0 = movement toward SCC\n"
        )

        handle.write(
            "Logit delta < 0 = movement toward BCC\n\n"
        )

        for row in summaries:
            handle.write(
                f"{row['group'].upper()}\n"
            )

            handle.write(
                "-" * 80 + "\n"
            )

            handle.write(
                f"N: {row['n']}\n"
            )

            handle.write(
                f"Pearson r: "
                f"{row['pearson_r']:.6f}\n"
            )

            handle.write(
                f"Pearson p: "
                f"{row['pearson_p']:.6f}\n"
            )

            handle.write(
                f"Spearman rho: "
                f"{row['spearman_rho']:.6f}\n"
            )

            handle.write(
                f"Spearman p: "
                f"{row['spearman_p']:.6f}\n"
            )

            handle.write(
                "Semantic-direction agreement: "
                f"{row['same_direction_fraction']:.6f}\n\n"
            )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        f"Lesion table: {lesion_output}"
    )

    print(
        f"Summary CSV:  {summary_output}"
    )

    print(
        f"Summary:      {report_output}"
    )

    print()
    print("=" * 80)
    print(
        "GEOMETRY → LOGIT ALIGNMENT "
        "ANALYSIS COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
