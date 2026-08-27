from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare C1 and F1 SCC lesion-level "
            "feature geometry."
        )
    )

    parser.add_argument(
        "--c1",
        default=(
            "analysis/scc_bcc/"
            "c1_lesion_geometry/"
            "scc_lesion_geometry.csv"
        ),
        help="C1 SCC lesion geometry CSV.",
    )

    parser.add_argument(
        "--f1",
        default=(
            "analysis/scc_bcc/"
            "f1/"
            "scc_lesion_geometry.csv"
        ),
        help="F1 SCC lesion geometry CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "analysis/scc_bcc/"
            "f1"
        ),
        help="Output directory.",
    )

    return parser.parse_args()


REQUIRED_BASE_COLUMNS = {
    "patient_id",
    "lesion_uid",
    "error_fraction",
    "distance_to_bcc_centroid",
    "distance_to_scc_centroid",
}


def load_geometry(
    path: Path,
    label: str,
):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} geometry file does not exist: "
            f"{path}"
        )

    df = pd.read_csv(path)

    missing = REQUIRED_BASE_COLUMNS - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{label} geometry file is missing "
            f"required columns: {sorted(missing)}"
        )

    # C1 uses the corrected field name.
    #
    # The existing F1 geometry file was generated before
    # the terminology correction and therefore contains
    # the legacy name "scc_advantage_over_bcc".
    #
    # The underlying quantity is the same:
    #
    #     distance_to_SCC - distance_to_BCC
    #
    # Positive -> closer to BCC
    # Negative -> closer to SCC
    if "bcc_minus_scc_distance" not in df.columns:
        if (
            label == "F1"
            and "scc_advantage_over_bcc" in df.columns
        ):
            df = df.rename(
                columns={
                    "scc_advantage_over_bcc":
                    "bcc_minus_scc_distance"
                }
            )
        else:
            raise RuntimeError(
                f"{label} geometry file is missing "
                "bcc_minus_scc_distance."
            )

    key = [
        "patient_id",
        "lesion_uid",
    ]

    if df[key].duplicated().any():
        duplicates = df.loc[
            df[key].duplicated(keep=False),
            key,
        ]

        raise RuntimeError(
            f"{label} contains duplicate lesion keys:\n"
            f"{duplicates.to_string(index=False)}"
        )

    return df


def main():
    args = parse_args()

    c1_path = Path(args.c1)
    f1_path = Path(args.f1)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("C1 → F1 PAIRED SCC LESION DISPLACEMENT")
    print("=" * 80)

    c1 = load_geometry(
        c1_path,
        "C1",
    )

    f1 = load_geometry(
        f1_path,
        "F1",
    )

    print()
    print(
        f"C1 lesions: {len(c1)}"
    )

    print(
        f"F1 lesions: {len(f1)}"
    )

    # ------------------------------------------------------------
    # Verify exact lesion identity alignment.
    # ------------------------------------------------------------

    key = [
        "patient_id",
        "lesion_uid",
    ]

    c1_keys = set(
        map(
            tuple,
            c1[key].to_numpy(),
        )
    )

    f1_keys = set(
        map(
            tuple,
            f1[key].to_numpy(),
        )
    )

    only_c1 = c1_keys - f1_keys
    only_f1 = f1_keys - c1_keys

    if only_c1:
        raise RuntimeError(
            "Lesions present in C1 but missing from F1:\n"
            f"{sorted(only_c1)}"
        )

    if only_f1:
        raise RuntimeError(
            "Lesions present in F1 but missing from C1:\n"
            f"{sorted(only_f1)}"
        )

    if c1_keys != f1_keys:
        raise RuntimeError(
            "C1/F1 lesion identity mismatch."
        )

    if len(c1_keys) != 22:
        raise RuntimeError(
            "Expected exactly 22 SCC lesions, got "
            f"{len(c1_keys)}."
        )

    # ------------------------------------------------------------
    # Select comparable columns.
    # ------------------------------------------------------------

    c1_compare = c1[
        key
        + [
            "error_fraction",
            "distance_to_bcc_centroid",
            "distance_to_scc_centroid",
            "bcc_minus_scc_distance",
        ]
    ].copy()

    f1_compare = f1[
        key
        + [
            "error_fraction",
            "distance_to_bcc_centroid",
            "distance_to_scc_centroid",
            "bcc_minus_scc_distance",
        ]
    ].copy()

    merged = c1_compare.merge(
        f1_compare,
        on=key,
        how="inner",
        suffixes=(
            "_c1",
            "_f1",
        ),
        validate="one_to_one",
    )

    if len(merged) != 22:
        raise RuntimeError(
            "Expected 22 matched lesions after merge, got "
            f"{len(merged)}."
        )

    # ------------------------------------------------------------
    # Verify problematic/clean grouping is identical.
    #
    # A lesion is problematic if at least one of its images
    # produced an SCC -> BCC error.
    # ------------------------------------------------------------

    c1_problematic = (
        merged["error_fraction_c1"] > 0
    )

    f1_problematic = (
        merged["error_fraction_f1"] > 0
    )

    if not np.array_equal(
        c1_problematic.to_numpy(),
        f1_problematic.to_numpy(),
    ):
        raise RuntimeError(
            "C1 and F1 error-status grouping differs. "
            "The F1 error CSV may have been generated "
            "with a different test prediction set."
        )

    merged["group"] = np.where(
        c1_problematic,
        "problematic",
        "clean",
    )

    # ------------------------------------------------------------
    # Compute paired displacement.
    #
    # Margin definition:
    #
    #   BCC-minus-SCC distance
    #     = distance_to_SCC - distance_to_BCC
    #
    # Positive -> BCC is closer.
    # Negative -> SCC is closer.
    #
    # Therefore:
    #
    #   negative Δ margin -> movement toward SCC
    #   positive Δ margin -> movement toward BCC
    # ------------------------------------------------------------

    merged["delta_margin"] = (
        merged["bcc_minus_scc_distance_f1"]
        - merged["bcc_minus_scc_distance_c1"]
    )

    merged["delta_bcc_distance"] = (
        merged["distance_to_bcc_centroid_f1"]
        - merged["distance_to_bcc_centroid_c1"]
    )

    merged["delta_scc_distance"] = (
        merged["distance_to_scc_centroid_f1"]
        - merged["distance_to_scc_centroid_c1"]
    )

    merged["c1_bcc_side"] = (
        merged["bcc_minus_scc_distance_c1"] > 0
    )

    merged["f1_bcc_side"] = (
        merged["bcc_minus_scc_distance_f1"] > 0
    )

    merged["c1_scc_side"] = (
        merged["bcc_minus_scc_distance_c1"] < 0
    )

    merged["f1_scc_side"] = (
        merged["bcc_minus_scc_distance_f1"] < 0
    )

    # ------------------------------------------------------------
    # Print group summaries.
    # ------------------------------------------------------------

    print()
    print(
        f"Matched lesions: {len(merged)}"
    )

    print(
        "Problematic:    "
        f"{int(c1_problematic.sum())}"
    )

    print(
        "Clean:          "
        f"{int((~c1_problematic).sum())}"
    )

    for group_name in (
        "problematic",
        "clean",
    ):
        group = merged[
            merged["group"] == group_name
        ]

        print()
        print(
            group_name.upper()
        )
        print("-" * 80)

        print(
            f"Mean C1 margin:       "
            f"{group['bcc_minus_scc_distance_c1'].mean():.6f}"
        )

        print(
            f"Mean F1 margin:       "
            f"{group['bcc_minus_scc_distance_f1'].mean():.6f}"
        )

        print(
            f"Mean Δ margin:        "
            f"{group['delta_margin'].mean():.6f}"
        )

        print(
            f"Median Δ margin:      "
            f"{group['delta_margin'].median():.6f}"
        )

        print(
            f"Mean Δ BCC distance:  "
            f"{group['delta_bcc_distance'].mean():.6f}"
        )

        print(
            f"Mean Δ SCC distance:  "
            f"{group['delta_scc_distance'].mean():.6f}"
        )

        print(
            f"C1 BCC-side lesions:  "
            f"{int(group['c1_bcc_side'].sum())}/"
            f"{len(group)}"
        )

        print(
            f"F1 BCC-side lesions:  "
            f"{int(group['f1_bcc_side'].sum())}/"
            f"{len(group)}"
        )

        print(
            f"C1 SCC-side lesions:  "
            f"{int(group['c1_scc_side'].sum())}/"
            f"{len(group)}"
        )

        print(
            f"F1 SCC-side lesions:  "
            f"{int(group['f1_scc_side'].sum())}/"
            f"{len(group)}"
        )

    # ------------------------------------------------------------
    # Paired statistical test.
    #
    # The paired Wilcoxon test asks whether the median lesion-level
    # change in BCC-minus-SCC margin differs from zero.
    #
    # This is exploratory and is NOT treated as an independent
    # confirmation of the earlier lesion-level analysis.
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("PAIRED WILCOXON TEST")
    print("=" * 80)

    delta = merged[
        "delta_margin"
    ].to_numpy()

    if np.allclose(delta, 0):
        print(
            "All paired margin changes are zero; "
            "Wilcoxon test not applicable."
        )

        wilcoxon_statistic = float("nan")
        wilcoxon_p = float("nan")
    else:
        wilcoxon_result = wilcoxon(
            delta,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        wilcoxon_statistic = float(
            wilcoxon_result.statistic
        )

        wilcoxon_p = float(
            wilcoxon_result.pvalue
        )

        print(
            f"Wilcoxon statistic: "
            f"{wilcoxon_statistic:.6f}"
        )

        print(
            f"p-value:            "
            f"{wilcoxon_p:.6f}"
        )

    # ------------------------------------------------------------
    # Overall movement.
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("ALL-LESION PAIRED CHANGE")
    print("=" * 80)

    print(
        f"Mean Δ margin:   "
        f"{merged['delta_margin'].mean():.6f}"
    )

    print(
        f"Median Δ margin: "
        f"{merged['delta_margin'].median():.6f}"
    )

    print(
        f"Mean Δ BCC distance: "
        f"{merged['delta_bcc_distance'].mean():.6f}"
    )

    print(
        f"Mean Δ SCC distance: "
        f"{merged['delta_scc_distance'].mean():.6f}"
    )

    print()
    print(
        "Negative Δ margin = movement toward SCC."
    )

    print(
        "Positive Δ margin = movement toward BCC."
    )

    # ------------------------------------------------------------
    # Largest movements.
    # ------------------------------------------------------------

    print()
    print(
        "Largest positive Δ margin "
        "(movement toward BCC):"
    )

    print(
        merged.nlargest(
            5,
            "delta_margin",
        )[
            key
            + [
                "group",
                "delta_margin",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Largest negative Δ margin "
        "(movement toward SCC):"
    )

    print(
        merged.nsmallest(
            5,
            "delta_margin",
        )[
            key
            + [
                "group",
                "delta_margin",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Save paired lesion table.
    # ------------------------------------------------------------

    csv_path = (
        output_dir
        / "c1_vs_f1_lesion_displacement.csv"
    )

    merged.to_csv(
        csv_path,
        index=False,
    )

    print()
    print(
        f"Saved table: {csv_path}"
    )

    # ------------------------------------------------------------
    # Save summary.
    # ------------------------------------------------------------

    summary_path = (
        output_dir
        / "c1_vs_f1_lesion_displacement_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "DERMASENSE C1 → F1 "
            "PAIRED SCC LESION DISPLACEMENT\n"
        )

        handle.write(
            "=" * 80 + "\n\n"
        )

        handle.write(
            "Margin definition:\n"
        )

        handle.write(
            "  BCC-minus-SCC distance = "
            "distance_to_SCC - distance_to_BCC\n"
        )

        handle.write(
            "  Positive = closer to BCC\n"
        )

        handle.write(
            "  Negative = closer to SCC\n"
        )

        handle.write(
            "  Negative Δ = movement toward SCC\n"
        )

        handle.write(
            "  Positive Δ = movement toward BCC\n\n"
        )

        handle.write(
            f"Matched lesions: {len(merged)}\n"
        )

        handle.write(
            f"Problematic: "
            f"{int(c1_problematic.sum())}\n"
        )

        handle.write(
            f"Clean: "
            f"{int((~c1_problematic).sum())}\n\n"
        )

        for group_name in (
            "problematic",
            "clean",
        ):
            group = merged[
                merged["group"] == group_name
            ]

            handle.write(
                f"{group_name.upper()}\n"
            )

            handle.write(
                "-" * 60 + "\n"
            )

            handle.write(
                f"Mean C1 margin: "
                f"{group['bcc_minus_scc_distance_c1'].mean():.6f}\n"
            )

            handle.write(
                f"Mean F1 margin: "
                f"{group['bcc_minus_scc_distance_f1'].mean():.6f}\n"
            )

            handle.write(
                f"Mean Δ margin: "
                f"{group['delta_margin'].mean():.6f}\n"
            )

            handle.write(
                f"Median Δ margin: "
                f"{group['delta_margin'].median():.6f}\n"
            )

            handle.write(
                f"Mean Δ BCC distance: "
                f"{group['delta_bcc_distance'].mean():.6f}\n"
            )

            handle.write(
                f"Mean Δ SCC distance: "
                f"{group['delta_scc_distance'].mean():.6f}\n"
            )

            handle.write(
                f"C1 BCC-side: "
                f"{int(group['c1_bcc_side'].sum())}/"
                f"{len(group)}\n"
            )

            handle.write(
                f"F1 BCC-side: "
                f"{int(group['f1_bcc_side'].sum())}/"
                f"{len(group)}\n"
            )

            handle.write(
                f"C1 SCC-side: "
                f"{int(group['c1_scc_side'].sum())}/"
                f"{len(group)}\n"
            )

            handle.write(
                f"F1 SCC-side: "
                f"{int(group['f1_scc_side'].sum())}/"
                f"{len(group)}\n\n"
            )

        handle.write(
            "PAIRED WILCOXON TEST\n"
        )

        handle.write(
            f"Statistic: "
            f"{wilcoxon_statistic}\n"
        )

        handle.write(
            f"p-value: "
            f"{wilcoxon_p}\n\n"
        )

        handle.write(
            "OVERALL\n"
        )

        handle.write(
            f"Mean Δ margin: "
            f"{merged['delta_margin'].mean():.6f}\n"
        )

        handle.write(
            f"Median Δ margin: "
            f"{merged['delta_margin'].median():.6f}\n"
        )

        handle.write(
            f"Mean Δ BCC distance: "
            f"{merged['delta_bcc_distance'].mean():.6f}\n"
        )

        handle.write(
            f"Mean Δ SCC distance: "
            f"{merged['delta_scc_distance'].mean():.6f}\n"
        )

    print(
        f"Saved summary: {summary_path}"
    )

    print()
    print("=" * 80)
    print("C1 → F1 LESION DISPLACEMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
