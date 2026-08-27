from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


INPUT_PATH = Path("analysis/product_eval/c1_f1_test_predictions.csv")
OUTPUT_DIR = Path("analysis/product_eval/phase4_safety_policy")

HIGH_RISK_CLASSES = {"BCC", "SCC", "MEL"}
LOW_RISK_CLASSES = {"NEV", "SEK"}

# ACK is intentionally intermediate:
# it is not treated as "monitor" and therefore is not a dangerous
# high-risk -> monitor downgrade.
EVALUATE_SOON_CLASSES = {"ACK"}

MODELS = {
    "C1": {
        "pred": "c1_pred",
        "confidence": "c1_confidence",
        "high_risk_probability": "c1_high_risk_probability",
    },
    "F1": {
        "pred": "f1_pred",
        "confidence": "f1_confidence",
        "high_risk_probability": "f1_high_risk_probability",
    },
}


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


def classify_action(predicted_class: str) -> str:
    if predicted_class in HIGH_RISK_CLASSES:
        return "HIGH_RISK"
    if predicted_class in EVALUATE_SOON_CLASSES:
        return "EVALUATE_SOON"
    if predicted_class in LOW_RISK_CLASSES:
        return "MONITOR"
    return "UNKNOWN"


def is_dangerous_downgrade(true_class: str, predicted_class: str) -> bool:
    """
    A dangerous failure occurs when a truly high-risk lesion is sent to
    the lowest-action product category: MONITOR.

    This deliberately measures downstream product consequence rather
    than merely native-class misclassification.
    """
    return (
        true_class in HIGH_RISK_CLASSES
        and classify_action(predicted_class) == "MONITOR"
    )


def policy_baseline(
    df: pd.DataFrame,
    pred_col: str,
    confidence_col: str,
    high_risk_probability_col: str,
) -> pd.Series:
    return pd.Series(False, index=df.index)


def policy_global_confidence(
    df: pd.DataFrame,
    pred_col: str,
    confidence_col: str,
    high_risk_probability_col: str,
    threshold: float,
) -> pd.Series:
    return df[confidence_col] < threshold


def policy_low_risk_review(
    df: pd.DataFrame,
    pred_col: str,
    confidence_col: str,
    high_risk_probability_col: str,
) -> pd.Series:
    return df[pred_col].isin(LOW_RISK_CLASSES)


def policy_low_risk_confidence(
    df: pd.DataFrame,
    pred_col: str,
    confidence_col: str,
    high_risk_probability_col: str,
    threshold: float,
) -> pd.Series:
    return (
        df[pred_col].isin(LOW_RISK_CLASSES)
        & (df[confidence_col] < threshold)
    )


def policy_high_risk_probability(
    df: pd.DataFrame,
    pred_col: str,
    confidence_col: str,
    high_risk_probability_col: str,
    threshold: float,
) -> pd.Series:
    """
    Review a native low-risk prediction when the model nevertheless
    assigns a meaningful probability to the high-risk bucket.
    """
    return (
        df[pred_col].isin(LOW_RISK_CLASSES)
        & (df[high_risk_probability_col] >= threshold)
    )


def evaluate_policy(
    df: pd.DataFrame,
    model_name: str,
    policy_name: str,
    review_mask: pd.Series,
    pred_col: str,
) -> dict:
    review_mask = review_mask.astype(bool)

    true_class = df["true_class"]

    dangerous = true_class.combine(
        df[pred_col],
        lambda t, p: is_dangerous_downgrade(t, p),
    )

    dangerous_reviewed = int((dangerous & review_mask).sum())
    dangerous_total = int(dangerous.sum())
    dangerous_remaining = int((dangerous & ~review_mask).sum())

    review_count = int(review_mask.sum())
    total = len(df)

    correct = df["true_class"].eq(df[pred_col])
    correct_sent_to_review = int((correct & review_mask).sum())

    incorrect_sent_to_review = int(
        ((~correct) & review_mask).sum()
    )

    review_rate = review_count / total if total else np.nan

    catch_rate = (
        dangerous_reviewed / dangerous_total
        if dangerous_total
        else np.nan
    )

    # "Safety-adjusted coverage" here means the fraction of cases that
    # can proceed without review while avoiding dangerous downgrades.
    auto_release = ~review_mask
    auto_release_count = int(auto_release.sum())

    safe_auto_release = int(
        (auto_release & ~dangerous).sum()
    )

    safe_auto_release_rate = (
        safe_auto_release / total if total else np.nan
    )

    return {
        "model": model_name,
        "policy": policy_name,
        "total_cases": total,
        "review_cases": review_count,
        "review_rate": review_rate,
        "correct_predictions_sent_to_review": correct_sent_to_review,
        "incorrect_predictions_sent_to_review": incorrect_sent_to_review,
        "dangerous_high_risk_to_monitor": dangerous_total,
        "dangerous_caught": dangerous_reviewed,
        "dangerous_catch_rate": catch_rate,
        "dangerous_remaining": dangerous_remaining,
        "safe_auto_release_rate": safe_auto_release_rate,
    }


def print_policy_table(results: list[dict], model_name: str) -> None:
    rows = [r for r in results if r["model"] == model_name]

    out = pd.DataFrame(rows)

    display_cols = [
        "policy",
        "review_rate",
        "dangerous_catch_rate",
        "dangerous_remaining",
        "correct_predictions_sent_to_review",
        "incorrect_predictions_sent_to_review",
        "safe_auto_release_rate",
    ]

    print(out[display_cols].to_string(index=False, formatters={
        "review_rate": lambda x: f"{x:.1%}",
        "dangerous_catch_rate": lambda x: f"{x:.1%}",
        "safe_auto_release_rate": lambda x: f"{x:.1%}",
    }))


def print_dangerous_cases(
    df: pd.DataFrame,
    model_name: str,
    pred_col: str,
    confidence_col: str,
    high_risk_probability_col: str,
) -> None:
    dangerous = df.apply(
        lambda row: is_dangerous_downgrade(
            row["true_class"],
            row[pred_col],
        ),
        axis=1,
    )

    cols = [
        "image_id",
        "true_class",
        pred_col,
        confidence_col,
        high_risk_probability_col,
    ]

    print()
    print(f"{model_name} DANGEROUS HIGH-RISK → MONITOR CASES")
    print("-" * 80)

    print(
        df.loc[dangerous, cols]
        .sort_values(confidence_col)
        .to_string(index=False)
    )


def build_policy_results(
    df: pd.DataFrame,
    model_name: str,
) -> list[dict]:
    config = MODELS[model_name]

    pred_col = config["pred"]
    confidence_col = config["confidence"]
    high_risk_probability_col = config["high_risk_probability"]

    results: list[dict] = []

    policies: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = []

    policies.append(
        (
            "baseline_no_gate",
            lambda x: policy_baseline(
                x,
                pred_col,
                confidence_col,
                high_risk_probability_col,
            ),
        )
    )

    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
        policies.append(
            (
                f"global_confidence_lt_{threshold:.2f}",
                lambda x, t=threshold: policy_global_confidence(
                    x,
                    pred_col,
                    confidence_col,
                    high_risk_probability_col,
                    t,
                ),
            )
        )

    policies.append(
        (
            "low_risk_prediction_review",
            lambda x: policy_low_risk_review(
                x,
                pred_col,
                confidence_col,
                high_risk_probability_col,
            ),
        )
    )

    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
        policies.append(
            (
                f"low_risk_confidence_lt_{threshold:.2f}",
                lambda x, t=threshold: policy_low_risk_confidence(
                    x,
                    pred_col,
                    confidence_col,
                    high_risk_probability_col,
                    t,
                ),
            )
        )

    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50]:
        policies.append(
            (
                f"low_risk_high_risk_prob_ge_{threshold:.2f}",
                lambda x, t=threshold: policy_high_risk_probability(
                    x,
                    pred_col,
                    confidence_col,
                    high_risk_probability_col,
                    t,
                ),
            )
        )

    for policy_name, policy_fn in policies:
        review_mask = policy_fn(df)

        results.append(
            evaluate_policy(
                df=df,
                model_name=model_name,
                policy_name=policy_name,
                review_mask=review_mask,
                pred_col=pred_col,
            )
        )

    return results


def main() -> None:
    print("=" * 80)
    print("DERMASENSE PHASE 4 SAFETY POLICY DESIGN")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prediction artifact not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = [
        "image_id",
        "true_class",
    ]

    for config in MODELS.values():
        required.extend([
            config["pred"],
            config["confidence"],
        ])

    require_columns(df, required)

    # The product-evaluation artifact stores the individual class
    # probabilities rather than a precomputed high-risk probability.
    #
    # DermaSense high-risk definition:
    #     BCC + MEL + SCC
    #
    # We derive this quantity here rather than modifying the original
    # prediction artifact.
    for model_name, config in MODELS.items():
        prefix = model_name.lower()

        probability_columns = [
            f"{prefix}_bcc_probability",
            f"{prefix}_mel_probability",
            f"{prefix}_scc_probability",
        ]

        require_columns(
            df,
            probability_columns,
        )

        df[config["high_risk_probability"]] = (
            df[probability_columns[0]]
            + df[probability_columns[1]]
            + df[probability_columns[2]]
        )

    print(f"Input: {INPUT_PATH}")
    print(f"Rows:  {len(df)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for model_name in MODELS:
        all_results.extend(
            build_policy_results(df, model_name)
        )

    results_df = pd.DataFrame(all_results)

    results_path = OUTPUT_DIR / "safety_policy_comparison.csv"
    results_df.to_csv(results_path, index=False)

    for model_name in MODELS:
        print()
        print("=" * 80)
        print(f"{model_name} SAFETY POLICY COMPARISON")
        print("=" * 80)

        print_policy_table(
            all_results,
            model_name,
        )

        config = MODELS[model_name]

        print_dangerous_cases(
            df,
            model_name,
            config["pred"],
            config["confidence"],
            config["high_risk_probability"],
        )

    summary_lines = [
        "DERMASENSE PHASE 4 SAFETY POLICY DESIGN",
        "=" * 80,
        "",
        "Purpose:",
        "Compare candidate safety-gate policies using downstream",
        "action consequences rather than ordinary classification accuracy.",
        "",
        "HIGH-RISK:",
        ", ".join(sorted(HIGH_RISK_CLASSES)),
        "",
        "LOW-RISK / MONITOR:",
        ", ".join(sorted(LOW_RISK_CLASSES)),
        "",
        "INTERMEDIATE / EVALUATE SOON:",
        ", ".join(sorted(EVALUATE_SOON_CLASSES)),
        "",
        "Dangerous failure definition:",
        "True high-risk class predicted into MONITOR.",
        "",
        "Candidate policies:",
        "1. baseline_no_gate",
        "2. global confidence thresholds",
        "3. low-risk prediction -> review",
        "4. low-risk + confidence -> review",
        "5. low-risk + high-risk probability -> review",
        "",
        "Important:",
        "This is a policy-design experiment, not production validation.",
        "No threshold or policy should be considered clinically validated",
        "from this test set alone.",
        "",
        f"Results: {results_path}",
        "",
    ]

    summary_path = OUTPUT_DIR / "safety_policy_design_summary.txt"
    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)
    print(f"Comparison: {results_path}")
    print(f"Summary:    {summary_path}")
    print()
    print("=" * 80)
    print("PHASE 4 SAFETY POLICY DESIGN COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
