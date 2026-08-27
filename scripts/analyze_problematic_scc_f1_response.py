from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze lesion-level F1 response among "
            "problematic SCC cases."
        )
    )

    parser.add_argument(
        "--geometry-logit",
        default=(
            "analysis/scc_bcc/geometry_vs_logits/"
            "geometry_vs_logits_lesions.csv"
        ),
        help="Geometry/logit alignment lesion table.",
    )

    parser.add_argument(
        "--clinical",
        default=(
            "analysis/scc_bcc/clinical_metadata/"
            "scc_lesion_clinical_metadata.csv"
        ),
        help="SCC clinical metadata table.",
    )

    parser.add_argument(
        "--displacement",
        default=(
            "analysis/scc_bcc/f1/"
            "c1_vs_f1_lesion_displacement.csv"
        ),
        help="C1 vs F1 lesion displacement table.",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "analysis/scc_bcc/"
            "problematic_f1_response"
        ),
        help="Output directory.",
    )

    return parser.parse_args()


def require_columns(df, columns, name):
    missing = set(columns) - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{name} missing columns: "
            f"{sorted(missing)}"
        )


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def classify_response(row):
    """
    Embedding convention:

        delta_embedding_margin < 0
            = movement toward SCC

        delta_embedding_margin > 0
            = movement toward BCC

    Logit convention:

        delta_logit_margin > 0
            = movement toward SCC

        delta_logit_margin < 0
            = movement toward BCC

    Therefore:

        A = both embedding and classifier improved
        B = embedding improved only
        C = both worsened
        D = classifier improved only
    """

    embedding_delta = safe_float(
        row["delta_embedding_margin"]
    )

    logit_delta = safe_float(
        row["delta_logit_margin"]
    )

    embedding_improved = (
        np.isfinite(embedding_delta)
        and embedding_delta < 0
    )

    logit_improved = (
        np.isfinite(logit_delta)
        and logit_delta > 0
    )

    if embedding_improved and logit_improved:
        return (
            "A_both_improved",
            embedding_improved,
            logit_improved,
        )

    if embedding_improved and not logit_improved:
        return (
            "B_embedding_only",
            embedding_improved,
            logit_improved,
        )

    if not embedding_improved and logit_improved:
        return (
            "D_logit_only",
            embedding_improved,
            logit_improved,
        )

    return (
        "C_both_worsened",
        embedding_improved,
        logit_improved,
    )


def print_group_summary(df, label):
    print()
    print("=" * 80)
    print(label)
    print("=" * 80)

    print(
        f"N lesions: {len(df)}"
    )

    if len(df) == 0:
        return

    for column, title in (
        (
            "diameter_1",
            "diameter_1",
        ),
        (
            "diameter_2",
            "diameter_2",
        ),
        (
            "lesion_area_proxy",
            "lesion_area_proxy",
        ),
        (
            "hurt",
            "hurt",
        ),
        (
            "skin_cancer_history",
            "skin_cancer_history",
        ),
    ):
        if column not in df.columns:
            continue

        if column in (
            "diameter_1",
            "diameter_2",
            "lesion_area_proxy",
        ):
            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).dropna()

            if len(values):
                print(
                    f"{title}: "
                    f"mean={values.mean():.4f} "
                    f"median={values.median():.4f}"
                )
        else:
            print()
            print(
                f"{title} distribution:"
            )

            print(
                df[column]
                .astype(str)
                .value_counts()
                .to_string()
            )


def print_response_summary(df):
    print()
    print("=" * 80)
    print("RESPONSE MAGNITUDE SUMMARY")
    print("=" * 80)

    for response_class in (
        "A_both_improved",
        "B_embedding_only",
        "C_both_worsened",
        "D_logit_only",
    ):
        subset = df[
            df["response_class"]
            == response_class
        ]

        print()
        print(response_class)
        print("-" * 80)

        print(
            f"N: {len(subset)}"
        )

        if len(subset) == 0:
            continue

        for column in (
            "delta_embedding_margin",
            "delta_logit_margin",
        ):
            values = pd.to_numeric(
                subset[column],
                errors="coerce",
            ).dropna()

            if len(values):
                print(
                    f"{column}: "
                    f"mean={values.mean():.6f} "
                    f"median={values.median():.6f}"
                )


def main():
    args = parse_args()

    geometry_logit_path = Path(
        args.geometry_logit
    )

    clinical_path = Path(
        args.clinical
    )

    displacement_path = Path(
        args.displacement
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
        "DERMASENSE PROBLEMATIC SCC "
        "F1 RESPONSE ANALYSIS"
    )
    print("=" * 80)

    for path in (
        geometry_logit_path,
        clinical_path,
        displacement_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    geometry_logit = pd.read_csv(
        geometry_logit_path
    )

    clinical = pd.read_csv(
        clinical_path
    )

    displacement = pd.read_csv(
        displacement_path
    )

    require_columns(
        geometry_logit,
        {
            "patient_id",
            "lesion_uid",
            "group",
            "delta_embedding_margin",
            "delta_logit_margin",
        },
        "Geometry/logit table",
    )

    require_columns(
        clinical,
        {
            "patient_id",
            "lesion_uid",
        },
        "Clinical table",
    )

    require_columns(
        displacement,
        {
            "patient_id",
            "lesion_uid",
            "error_fraction_c1",
            "error_fraction_f1",
        },
        "Displacement table",
    )

    # ------------------------------------------------------------
    # Keep only the clinical variables we need.
    # ------------------------------------------------------------

    clinical_columns = [
        "patient_id",
        "lesion_uid",
        "age",
        "diameter_1",
        "diameter_2",
        "fitspatrick",
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
    ]

    clinical_columns = [
        column
        for column in clinical_columns
        if column in clinical.columns
    ]

    clinical_subset = clinical[
        clinical_columns
    ].copy()

    # ------------------------------------------------------------
    # Add lesion-level area proxy.
    # ------------------------------------------------------------

    if (
        "diameter_1"
        in clinical_subset.columns
        and "diameter_2"
        in clinical_subset.columns
    ):
        d1 = pd.to_numeric(
            clinical_subset["diameter_1"],
            errors="coerce",
        )

        d2 = pd.to_numeric(
            clinical_subset["diameter_2"],
            errors="coerce",
        )

        clinical_subset[
            "lesion_area_proxy"
        ] = d1 * d2

    # ------------------------------------------------------------
    # Validate uniqueness before merging.
    # ------------------------------------------------------------

    geometry_keys = geometry_logit[
        [
            "patient_id",
            "lesion_uid",
        ]
    ]

    if geometry_keys.duplicated().any():
        duplicates = geometry_logit.loc[
            geometry_keys.duplicated(
                keep=False
            ),
            [
                "patient_id",
                "lesion_uid",
            ],
        ]

        raise RuntimeError(
            "Geometry/logit table contains "
            "duplicate lesion keys:\n"
            f"{duplicates.to_string(index=False)}"
        )

    clinical_keys = clinical_subset[
        [
            "patient_id",
            "lesion_uid",
        ]
    ]

    if clinical_keys.duplicated().any():
        duplicates = clinical_subset.loc[
            clinical_keys.duplicated(
                keep=False
            ),
            [
                "patient_id",
                "lesion_uid",
            ],
        ]

        raise RuntimeError(
            "Clinical table contains "
            "duplicate lesion keys:\n"
            f"{duplicates.to_string(index=False)}"
        )

    displacement_keys = displacement[
        [
            "patient_id",
            "lesion_uid",
        ]
    ]

    if displacement_keys.duplicated().any():
        duplicates = displacement.loc[
            displacement_keys.duplicated(
                keep=False
            ),
            [
                "patient_id",
                "lesion_uid",
            ],
        ]

        raise RuntimeError(
            "Displacement table contains "
            "duplicate lesion keys:\n"
            f"{duplicates.to_string(index=False)}"
        )

    # ------------------------------------------------------------
    # Merge geometry/logit and clinical metadata.
    # ------------------------------------------------------------

    base = geometry_logit.merge(
        clinical_subset,
        on=[
            "patient_id",
            "lesion_uid",
        ],
        how="left",
        validate="one_to_one",
    )

    base = base.merge(
        displacement[
            [
                "patient_id",
                "lesion_uid",
                "error_fraction_c1",
                "error_fraction_f1",
            ]
        ],
        on=[
            "patient_id",
            "lesion_uid",
        ],
        how="left",
        validate="one_to_one",
    )

    if len(base) != 22:
        raise RuntimeError(
            "Expected 22 matched lesions after "
            f"merge, got {len(base)}."
        )

    if base[
        "error_fraction_c1"
    ].isna().any():
        missing = base.loc[
            base["error_fraction_c1"].isna(),
            [
                "patient_id",
                "lesion_uid",
            ],
        ]

        raise RuntimeError(
            "Some lesions are missing C1 "
            "error fractions:\n"
            f"{missing.to_string(index=False)}"
        )

    # ------------------------------------------------------------
    # Restrict to problematic SCC lesions.
    # ------------------------------------------------------------

    problematic = base[
        base["group"] == "problematic"
    ].copy()

    if len(problematic) != 11:
        raise RuntimeError(
            "Expected 11 problematic SCC "
            f"lesions, got {len(problematic)}."
        )

    # ------------------------------------------------------------
    # Classify each lesion.
    # ------------------------------------------------------------

    classifications = problematic.apply(
        classify_response,
        axis=1,
        result_type="expand",
    )

    classifications.columns = [
        "response_class",
        "embedding_improved",
        "logit_improved",
    ]

    problematic = pd.concat(
        [
            problematic.reset_index(drop=True),
            classifications.reset_index(drop=True),
        ],
        axis=1,
    )

    # ------------------------------------------------------------
    # Derived indicators.
    # ------------------------------------------------------------

    problematic[
        "embedding_moved_toward_scc"
    ] = (
        problematic[
            "delta_embedding_margin"
        ]
        < 0
    )

    problematic[
        "classifier_moved_toward_scc"
    ] = (
        problematic[
            "delta_logit_margin"
        ]
        > 0
    )

    problematic[
        "embedding_magnitude"
    ] = (
        pd.to_numeric(
            problematic[
                "delta_embedding_margin"
            ],
            errors="coerce",
        ).abs()
    )

    problematic[
        "logit_magnitude"
    ] = (
        pd.to_numeric(
            problematic[
                "delta_logit_margin"
            ],
            errors="coerce",
        ).abs()
    )

    # ------------------------------------------------------------
    # Response classification.
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("RESPONSE CLASSIFICATION")
    print("=" * 80)

    counts = (
        problematic[
            "response_class"
        ]
        .value_counts()
        .sort_index()
    )

    for response_class in (
        "A_both_improved",
        "B_embedding_only",
        "C_both_worsened",
        "D_logit_only",
    ):
        print(
            f"{response_class}: "
            f"{counts.get(response_class, 0)}"
        )

    # ------------------------------------------------------------
    # Directional counts.
    # ------------------------------------------------------------

    embedding_toward_scc = (
        problematic[
            "embedding_moved_toward_scc"
        ]
    )

    classifier_toward_scc = (
        problematic[
            "classifier_moved_toward_scc"
        ]
    )

    print()
    print("=" * 80)
    print("DIRECTIONAL RESPONSE")
    print("=" * 80)

    print(
        "Embedding movement toward SCC: "
        f"{int(embedding_toward_scc.sum())}/"
        f"{len(problematic)}"
    )

    print(
        "Classifier movement toward SCC: "
        f"{int(classifier_toward_scc.sum())}/"
        f"{len(problematic)}"
    )

    # ------------------------------------------------------------
    # Important interpretation:
    #
    # delta_logit_margin alone cannot establish whether a lesion
    # crossed the absolute classifier decision boundary.
    #
    # The current geometry/logit alignment table contains the
    # change in logit margin, not the absolute C1/F1 logit margins.
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("CLASSIFIER RESCUE INTERPRETATION")
    print("=" * 80)

    print(
        "Problematic SCC lesions with classifier "
        "movement toward SCC:"
    )

    print(
        f"{int(classifier_toward_scc.sum())}/"
        f"{len(problematic)}"
    )

    print()
    print(
        "Actual BCC-side → SCC-side classifier "
        "boundary crossing:"
    )

    print(
        "NOT DETERMINABLE from the current "
        "lesion-level alignment table."
    )

    print(
        "Reason: the table contains delta_logit_margin "
        "but not the absolute C1/F1 classifier margins."
    )

    # ------------------------------------------------------------
    # Detailed lesion table.
    # ------------------------------------------------------------

    display_columns = [
        "patient_id",
        "lesion_uid",
        "response_class",
        "delta_embedding_margin",
        "delta_logit_margin",
        "embedding_moved_toward_scc",
        "classifier_moved_toward_scc",
        "error_fraction_c1",
        "error_fraction_f1",
        "age",
        "diameter_1",
        "diameter_2",
        "lesion_area_proxy",
        "hurt",
        "skin_cancer_history",
        "region",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in problematic.columns
    ]

    print()
    print("=" * 80)
    print("PROBLEMATIC LESION TABLE")
    print("=" * 80)

    print(
        problematic[
            display_columns
        ]
        .sort_values(
            [
                "response_class",
                "delta_embedding_margin",
            ]
        )
        .to_string(index=False)
    )

    # ------------------------------------------------------------
    # Response magnitude summaries.
    # ------------------------------------------------------------

    print_response_summary(
        problematic
    )

    # ------------------------------------------------------------
    # Clinical summaries by response class.
    # ------------------------------------------------------------

    for response_class in (
        "A_both_improved",
        "B_embedding_only",
        "C_both_worsened",
        "D_logit_only",
    ):
        subset = problematic[
            problematic[
                "response_class"
            ] == response_class
        ]

        print_group_summary(
            subset,
            response_class,
        )

    # ------------------------------------------------------------
    # Save lesion-level table.
    # ------------------------------------------------------------

    lesion_output = (
        output_dir
        / "problematic_scc_f1_response.csv"
    )

    problematic.to_csv(
        lesion_output,
        index=False,
    )

    # ------------------------------------------------------------
    # Build response-class summary.
    # ------------------------------------------------------------

    summary_rows = []

    for response_class in (
        "A_both_improved",
        "B_embedding_only",
        "C_both_worsened",
        "D_logit_only",
    ):
        subset = problematic[
            problematic[
                "response_class"
            ] == response_class
        ]

        row = {
            "response_class": response_class,
            "n": len(subset),
        }

        for column in (
            "delta_embedding_margin",
            "delta_logit_margin",
            "error_fraction_c1",
            "error_fraction_f1",
            "diameter_1",
            "diameter_2",
            "lesion_area_proxy",
            "age",
        ):
            if column not in subset.columns:
                continue

            values = pd.to_numeric(
                subset[column],
                errors="coerce",
            ).dropna()

            row[
                f"{column}_mean"
            ] = (
                values.mean()
                if len(values)
                else np.nan
            )

            row[
                f"{column}_median"
            ] = (
                values.median()
                if len(values)
                else np.nan
            )

        if len(subset):
            row[
                "embedding_toward_scc_n"
            ] = int(
                subset[
                    "embedding_moved_toward_scc"
                ].sum()
            )

            row[
                "classifier_toward_scc_n"
            ] = int(
                subset[
                    "classifier_moved_toward_scc"
                ].sum()
            )
        else:
            row[
                "embedding_toward_scc_n"
            ] = 0

            row[
                "classifier_toward_scc_n"
            ] = 0

        summary_rows.append(row)

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_output = (
        output_dir
        / "response_class_summary.csv"
    )

    summary_df.to_csv(
        summary_output,
        index=False,
    )

    # ------------------------------------------------------------
    # Human-readable report.
    # ------------------------------------------------------------

    report_output = (
        output_dir
        / "summary.txt"
    )

    with report_output.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "DERMASENSE PROBLEMATIC SCC "
            "F1 RESPONSE ANALYSIS\n"
        )

        handle.write(
            "=" * 80 + "\n\n"
        )

        handle.write(
            "Matched SCC lesions: 22\n"
        )

        handle.write(
            "Problematic SCC lesions: 11\n\n"
        )

        handle.write(
            "DIRECTION CONVENTIONS\n"
        )

        handle.write(
            "-" * 80 + "\n"
        )

        handle.write(
            "Embedding delta < 0 = movement "
            "toward SCC\n"
        )

        handle.write(
            "Embedding delta > 0 = movement "
            "toward BCC\n"
        )

        handle.write(
            "Logit delta > 0 = movement "
            "toward SCC\n"
        )

        handle.write(
            "Logit delta < 0 = movement "
            "toward BCC\n\n"
        )

        handle.write(
            "RESPONSE COUNTS\n"
        )

        handle.write(
            "-" * 80 + "\n"
        )

        for response_class in (
            "A_both_improved",
            "B_embedding_only",
            "C_both_worsened",
            "D_logit_only",
        ):
            handle.write(
                f"{response_class}: "
                f"{counts.get(response_class, 0)}\n"
            )

        handle.write("\n")

        handle.write(
            "DIRECTIONAL RESPONSE\n"
        )

        handle.write(
            "-" * 80 + "\n"
        )

        handle.write(
            "Embedding movement toward SCC: "
            f"{int(embedding_toward_scc.sum())}/"
            f"{len(problematic)}\n"
        )

        handle.write(
            "Classifier movement toward SCC: "
            f"{int(classifier_toward_scc.sum())}/"
            f"{len(problematic)}\n\n"
        )

        handle.write(
            "CLASSIFIER RESCUE\n"
        )

        handle.write(
            "-" * 80 + "\n"
        )

        handle.write(
            "Actual BCC-side → SCC-side "
            "classifier boundary crossing: "
            "NOT DETERMINABLE\n"
        )

        handle.write(
            "Reason: this analysis table contains "
            "delta_logit_margin but not absolute "
            "C1/F1 classifier margins.\n\n"
        )

        handle.write(
            "DETAILED LESIONS\n"
        )

        handle.write(
            "-" * 80 + "\n"
        )

        handle.write(
            problematic[
                display_columns
            ]
            .sort_values(
                [
                    "response_class",
                    "delta_embedding_margin",
                ]
            )
            .to_string(index=False)
        )

        handle.write("\n")

    # ------------------------------------------------------------
    # Final status.
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        f"Lesion table: {lesion_output}"
    )

    print(
        f"Class summary: {summary_output}"
    )

    print(
        f"Report: {report_output}"
    )

    print()
    print("=" * 80)
    print(
        "PROBLEMATIC SCC RESPONSE "
        "ANALYSIS COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()