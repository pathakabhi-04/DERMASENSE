from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "analysis/product_eval/c1_f1_test_predictions.csv"
)

OUTPUT_DIR = Path(
    "analysis/product_eval/phase4_safety_bottleneck"
)


HIGH_RISK = {"BCC", "MEL", "SCC"}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def normalize_class(value):
    return str(value).strip().upper()


def confidence_column(df: pd.DataFrame, model: str) -> str:
    """
    Recover the model confidence column.

    Preferred:
        c1_confidence / f1_confidence

    Fallback:
        maximum probability among model probability columns.
    """

    direct_candidates = [
        f"{model}_confidence",
        f"{model}_prediction_confidence",
        f"{model}_max_probability",
        f"{model}_max_prob",
    ]

    for col in direct_candidates:
        if col in df.columns:
            return col

    probability_prefix = f"{model}_"

    probability_columns = [
        col
        for col in df.columns
        if col.startswith(probability_prefix)
        and (
            col.endswith("_probability")
            or col.endswith("_prob")
            or col.endswith("_proba")
        )
    ]

    if probability_columns:
        df[f"{model}_derived_confidence"] = df[
            probability_columns
        ].astype(float).max(axis=1)

        return f"{model}_derived_confidence"

    raise RuntimeError(
        f"Could not recover confidence for {model}.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


def get_prediction_column(df: pd.DataFrame, model: str) -> str:
    candidates = [
        f"{model}_pred",
        f"{model}_prediction",
        f"{model}_predicted_class",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise RuntimeError(
        f"Could not find prediction column for {model}.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


def get_true_column(df: pd.DataFrame) -> str:
    candidates = [
        "true_class",
        "target_class",
        "true_label",
        "native_diagnosis",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise RuntimeError(
        "Could not find true-class column.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


def classify_error(true_class, predicted_class):
    true_class = normalize_class(true_class)
    predicted_class = normalize_class(predicted_class)

    if true_class == predicted_class:
        return "CORRECT"

    if true_class in HIGH_RISK:
        if predicted_class not in HIGH_RISK:
            return "TIER_1"

        return "TIER_2"

    if predicted_class in HIGH_RISK:
        return "TIER_3"

    return "TIER_4"


def safe_mean(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def safe_median(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan

    return float(values.median())


def high_confidence_fraction(series, threshold):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan

    return float(
        (values >= threshold).mean()
    )


def summarize_confidence(df, confidence_col):
    return {
        "n": int(len(df)),
        "mean_confidence": safe_mean(
            df[confidence_col]
        ),
        "median_confidence": safe_median(
            df[confidence_col]
        ),
        "confidence_ge_0.70": high_confidence_fraction(
            df[confidence_col],
            0.70,
        ),
        "confidence_ge_0.80": high_confidence_fraction(
            df[confidence_col],
            0.80,
        ),
        "confidence_ge_0.90": high_confidence_fraction(
            df[confidence_col],
            0.90,
        ),
    }


# ---------------------------------------------------------------------
# Main model analysis
# ---------------------------------------------------------------------

def analyze_model(
    df: pd.DataFrame,
    model: str,
):
    pred_col = get_prediction_column(
        df,
        model,
    )

    conf_col = confidence_column(
        df,
        model,
    )

    working = df.copy()

    working["true_class"] = (
        working["__true_class"]
        .map(normalize_class)
    )

    working["predicted_class"] = (
        working[pred_col]
        .map(normalize_class)
    )

    working["error_tier"] = [
        classify_error(
            true,
            pred,
        )
        for true, pred in zip(
            working["true_class"],
            working["predicted_class"],
        )
    ]

    working["confidence"] = pd.to_numeric(
        working[conf_col],
        errors="coerce",
    )

    tier1 = working[
        working["error_tier"] == "TIER_1"
    ].copy()

    # -------------------------------------------------------------
    # Source-class contribution
    # -------------------------------------------------------------

    source = (
        tier1
        .groupby("true_class")
        .size()
        .rename("tier1_count")
        .reset_index()
    )

    total_tier1 = len(tier1)

    source["fraction_of_tier1"] = (
        source["tier1_count"]
        / total_tier1
        if total_tier1 > 0
        else np.nan
    )

    source = source.sort_values(
        "tier1_count",
        ascending=False,
    )

    # -------------------------------------------------------------
    # Source → destination breakdown
    # -------------------------------------------------------------

    destinations = (
        tier1
        .groupby(
            [
                "true_class",
                "predicted_class",
            ]
        )
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            "count",
            ascending=False,
        )
    )

    # -------------------------------------------------------------
    # Confidence by error type
    # -------------------------------------------------------------

    confidence_groups = []

    group_definitions = [
        (
            "CORRECT",
            working[
                working["error_tier"] == "CORRECT"
            ],
        ),
        (
            "ALL_INCORRECT",
            working[
                working["error_tier"] != "CORRECT"
            ],
        ),
        (
            "TIER_1",
            working[
                working["error_tier"] == "TIER_1"
            ],
        ),
        (
            "TIER_2",
            working[
                working["error_tier"] == "TIER_2"
            ],
        ),
        (
            "TIER_3",
            working[
                working["error_tier"] == "TIER_3"
            ],
        ),
        (
            "TIER_4",
            working[
                working["error_tier"] == "TIER_4"
            ],
        ),
    ]

    for name, subset in group_definitions:
        row = summarize_confidence(
            subset,
            "confidence",
        )

        row["group"] = name
        confidence_groups.append(row)

    confidence_summary = pd.DataFrame(
        confidence_groups
    )[
        [
            "group",
            "n",
            "mean_confidence",
            "median_confidence",
            "confidence_ge_0.70",
            "confidence_ge_0.80",
            "confidence_ge_0.90",
        ]
    ]

    # -------------------------------------------------------------
    # Tier-1 confidence by source
    # -------------------------------------------------------------

    tier1_confidence = []

    for source_class, subset in tier1.groupby(
        "true_class"
    ):
        row = summarize_confidence(
            subset,
            "confidence",
        )

        row["true_class"] = source_class
        tier1_confidence.append(row)

    if tier1_confidence:
        tier1_confidence = pd.DataFrame(
            tier1_confidence
        )[
            [
                "true_class",
                "n",
                "mean_confidence",
                "median_confidence",
                "confidence_ge_0.70",
                "confidence_ge_0.80",
                "confidence_ge_0.90",
            ]
        ].sort_values(
            "n",
            ascending=False,
        )
    else:
        tier1_confidence = pd.DataFrame(
            columns=[
                "true_class",
                "n",
                "mean_confidence",
                "median_confidence",
                "confidence_ge_0.70",
                "confidence_ge_0.80",
                "confidence_ge_0.90",
            ]
        )

    return (
        working,
        tier1,
        source,
        destinations,
        confidence_summary,
        tier1_confidence,
    )


# ---------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------

def make_recommendation(
    c1_confidence,
    f1_confidence,
):
    c1 = c1_confidence[
        c1_confidence["group"] == "TIER_1"
    ]

    f1 = f1_confidence[
        f1_confidence["group"] == "TIER_1"
    ]

    if len(c1) == 0 or len(f1) == 0:
        return (
            "Insufficient Tier-1 confidence data "
            "for a recommendation."
        )

    c1_mean = float(
        c1.iloc[0]["mean_confidence"]
    )

    f1_mean = float(
        f1.iloc[0]["mean_confidence"]
    )

    c1_high = float(
        c1.iloc[0]["confidence_ge_0.90"]
    )

    f1_high = float(
        f1.iloc[0]["confidence_ge_0.90"]
    )

    pooled_mean = np.nanmean(
        [c1_mean, f1_mean]
    )

    pooled_high = np.nanmean(
        [c1_high, f1_high]
    )

    lines = []

    lines.append(
        "PHASE 4 INTERVENTION INTERPRETATION"
    )
    lines.append(
        "-" * 80
    )

    lines.append(
        f"C1 Tier-1 mean confidence: {c1_mean:.4f}"
    )

    lines.append(
        f"F1 Tier-1 mean confidence: {f1_mean:.4f}"
    )

    lines.append(
        f"C1 Tier-1 confidence >= 0.90: "
        f"{c1_high:.1%}"
    )

    lines.append(
        f"F1 Tier-1 confidence >= 0.90: "
        f"{f1_high:.1%}"
    )

    lines.append("")

    if (
        pooled_mean < 0.70
        and pooled_high < 0.20
    ):
        lines.append(
            "INTERPRETATION: Tier-1 errors are "
            "predominantly low-confidence."
        )
        lines.append(
            "A confidence-based abstention/review "
            "mechanism is a strong candidate."
        )

    elif (
        pooled_mean >= 0.80
        or pooled_high >= 0.50
    ):
        lines.append(
            "INTERPRETATION: A substantial fraction "
            "of Tier-1 errors are high-confidence."
        )
        lines.append(
            "Confidence-only abstention is unlikely "
            "to solve the safety problem."
        )
        lines.append(
            "Prioritize a safety override, improved "
            "classification, or mandatory review logic."
        )

    else:
        lines.append(
            "INTERPRETATION: Tier-1 errors occupy a "
            "mixed-confidence regime."
        )
        lines.append(
            "Confidence gating may recover some errors, "
            "but should be evaluated as a safety layer "
            "rather than assumed to solve the problem."
        )

    lines.append("")
    lines.append(
        "IMPORTANT: This is an engineering screening "
        "interpretation, not a final intervention decision."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            INPUT_PATH
        )

    print("=" * 80)
    print(
        "DERMASENSE PHASE 4 SAFETY BOTTLENECK ANALYSIS"
    )
    print("=" * 80)

    print(
        f"Input: {INPUT_PATH}"
    )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Rows: {len(df)}"
    )

    true_col = get_true_column(df)

    df["__true_class"] = (
        df[true_col]
        .map(normalize_class)
    )

    print(
        f"True class column: {true_col}"
    )

    # -------------------------------------------------------------
    # C1
    # -------------------------------------------------------------

    (
        c1_working,
        c1_tier1,
        c1_source,
        c1_destinations,
        c1_confidence,
        c1_tier1_confidence,
    ) = analyze_model(
        df,
        "c1",
    )

    # -------------------------------------------------------------
    # F1
    # -------------------------------------------------------------

    (
        f1_working,
        f1_tier1,
        f1_source,
        f1_destinations,
        f1_confidence,
        f1_tier1_confidence,
    ) = analyze_model(
        df,
        "f1",
    )

    # -------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("TIER-1 SOURCE CLASS CONTRIBUTION")
    print("=" * 80)

    for model, source in [
        ("C1", c1_source),
        ("F1", f1_source),
    ]:
        print()
        print(model)
        print("-" * 80)

        if len(source) == 0:
            print("No Tier-1 errors.")
            continue

        for _, row in source.iterrows():
            print(
                f"{row['true_class']}: "
                f"{int(row['tier1_count'])} "
                f"({row['fraction_of_tier1']:.1%})"
            )

    print()
    print("=" * 80)
    print("TIER-1 DESTINATIONS")
    print("=" * 80)

    for model, destinations in [
        ("C1", c1_destinations),
        ("F1", f1_destinations),
    ]:
        print()
        print(model)
        print("-" * 80)

        if len(destinations) == 0:
            print("No Tier-1 errors.")
            continue

        for _, row in destinations.iterrows():
            print(
                f"{row['true_class']} → "
                f"{row['predicted_class']}: "
                f"{int(row['count'])}"
            )

    print()
    print("=" * 80)
    print("CONFIDENCE BY ERROR TYPE")
    print("=" * 80)

    for model, confidence in [
        ("C1", c1_confidence),
        ("F1", f1_confidence),
    ]:
        print()
        print(model)
        print("-" * 80)

        for _, row in confidence.iterrows():
            print(
                f"{row['group']:<15} "
                f"N={int(row['n']):3d} "
                f"mean={row['mean_confidence']:.4f} "
                f"median={row['median_confidence']:.4f} "
                f">=0.90={row['confidence_ge_0.90']:.1%}"
            )

    print()
    print("=" * 80)
    print("TIER-1 CONFIDENCE BY SOURCE CLASS")
    print("=" * 80)

    for model, confidence in [
        ("C1", c1_tier1_confidence),
        ("F1", f1_tier1_confidence),
    ]:
        print()
        print(model)
        print("-" * 80)

        for _, row in confidence.iterrows():
            print(
                f"{row['true_class']}: "
                f"N={int(row['n'])} "
                f"mean={row['mean_confidence']:.4f} "
                f"median={row['median_confidence']:.4f} "
                f">=0.90={row['confidence_ge_0.90']:.1%}"
            )

    # -------------------------------------------------------------
    # Recommendation
    # -------------------------------------------------------------

    recommendation = make_recommendation(
        c1_confidence,
        f1_confidence,
    )

    print()
    print("=" * 80)
    print(recommendation)

    # -------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------

    c1_source.to_csv(
        OUTPUT_DIR / "c1_tier1_source_classes.csv",
        index=False,
    )

    f1_source.to_csv(
        OUTPUT_DIR / "f1_tier1_source_classes.csv",
        index=False,
    )

    c1_destinations.to_csv(
        OUTPUT_DIR / "c1_tier1_destinations.csv",
        index=False,
    )

    f1_destinations.to_csv(
        OUTPUT_DIR / "f1_tier1_destinations.csv",
        index=False,
    )

    c1_confidence.to_csv(
        OUTPUT_DIR / "c1_confidence_by_error_type.csv",
        index=False,
    )

    f1_confidence.to_csv(
        OUTPUT_DIR / "f1_confidence_by_error_type.csv",
        index=False,
    )

    c1_tier1_confidence.to_csv(
        OUTPUT_DIR / "c1_tier1_confidence_by_source.csv",
        index=False,
    )

    f1_tier1_confidence.to_csv(
        OUTPUT_DIR / "f1_tier1_confidence_by_source.csv",
        index=False,
    )

    # Save row-level Tier-1 cases for inspection.
    c1_tier1.to_csv(
        OUTPUT_DIR / "c1_tier1_cases.csv",
        index=False,
    )

    f1_tier1.to_csv(
        OUTPUT_DIR / "f1_tier1_cases.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "phase4_bottleneck_summary.txt",
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "DERMASENSE PHASE 4 SAFETY BOTTLENECK ANALYSIS\n"
        )
        handle.write("=" * 80 + "\n\n")

        handle.write(
            f"Input: {INPUT_PATH}\n"
        )

        handle.write(
            f"Total test images: {len(df)}\n\n"
        )

        handle.write(
            "C1 TIER-1 SOURCE CLASSES\n"
        )
        handle.write("-" * 80 + "\n")
        handle.write(
            c1_source.to_string(index=False)
            + "\n\n"
        )

        handle.write(
            "F1 TIER-1 SOURCE CLASSES\n"
        )
        handle.write("-" * 80 + "\n")
        handle.write(
            f1_source.to_string(index=False)
            + "\n\n"
        )

        handle.write(
            "C1 TIER-1 DESTINATIONS\n"
        )
        handle.write("-" * 80 + "\n")
        handle.write(
            c1_destinations.to_string(index=False)
            + "\n\n"
        )

        handle.write(
            "F1 TIER-1 DESTINATIONS\n"
        )
        handle.write("-" * 80 + "\n")
        handle.write(
            f1_destinations.to_string(index=False)
            + "\n\n"
        )

        handle.write(
            "C1 CONFIDENCE BY ERROR TYPE\n"
        )
        handle.write("-" * 80 + "\n")
        handle.write(
            c1_confidence.to_string(index=False)
            + "\n\n"
        )

        handle.write(
            "F1 CONFIDENCE BY ERROR TYPE\n"
        )
        handle.write("-" * 80 + "\n")
        handle.write(
            f1_confidence.to_string(index=False)
            + "\n\n"
        )

        handle.write(
            "C1 TIER-1 CONFIDENCE BY SOURCE\n"
        )
        handle.write("-" * 80 + "\n")
        handle.write(
            c1_tier1_confidence.to_string(index=False)
            + "\n\n"
        )

        handle.write(
            "F1 TIER-1 CONFIDENCE BY SOURCE\n"
        )
        handle.write("-" * 80 + "\n")
        handle.write(
            f1_tier1_confidence.to_string(index=False)
            + "\n\n"
        )

        handle.write(
            recommendation
            + "\n"
        )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print()
    print(
        "PHASE 4 SAFETY BOTTLENECK ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
