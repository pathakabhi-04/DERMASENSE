from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "analysis/product_eval/c1_f1_test_predictions.csv"
)

OUTPUT_DIR = Path(
    "analysis/product_eval/"
    "phase4_safety_gate"
)

HIGH_RISK = {"BCC", "MEL", "SCC"}
LOW_RISK = {"NEV", "SEK"}

CONFIDENCE_THRESHOLDS = (
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def load_predictions() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = {
        "true_class",
        "c1_pred",
        "f1_pred",
        "c1_confidence",
        "f1_confidence",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    return df


def add_safety_columns(
    df: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    result = df.copy()

    pred_col = f"{model}_pred"

    result["true_high_risk"] = (
        result["true_class"].isin(HIGH_RISK)
    )

    result["predicted_high_risk"] = (
        result[pred_col].isin(HIGH_RISK)
    )

    result["predicted_low_risk"] = (
        result[pred_col].isin(LOW_RISK)
    )

    result["high_risk_to_monitor"] = (
        result["true_high_risk"]
        & result["predicted_low_risk"]
    )

    result["high_risk_to_evaluate_soon"] = (
        result["true_high_risk"]
        & (result[pred_col] == "ACK")
    )

    return result


def find_probability_columns(
    df: pd.DataFrame,
    model: str,
) -> dict[str, str]:
    found = {}

    for class_name in (
        "ACK",
        "BCC",
        "MEL",
        "NEV",
        "SCC",
        "SEK",
    ):
        candidates = (
            f"{model}_{class_name.lower()}_probability",
            f"{model}_{class_name.lower()}_prob",
            f"{model}_{class_name}_probability",
            f"{model}_{class_name}_prob",
        )

        for candidate in candidates:
            if candidate in df.columns:
                found[class_name] = candidate
                break

    return found


def add_high_risk_probability(
    df: pd.DataFrame,
    model: str,
) -> tuple[pd.DataFrame, bool]:
    result = df.copy()

    probability_columns = find_probability_columns(
        result,
        model,
    )

    required = {
        "BCC",
        "MEL",
        "SCC",
    }

    if not required.issubset(probability_columns):
        return result, False

    result["predicted_high_risk_probability"] = (
        result[probability_columns["BCC"]]
        + result[probability_columns["MEL"]]
        + result[probability_columns["SCC"]]
    )

    return result, True


def evaluate_gate(
    df: pd.DataFrame,
    model: str,
    gate_name: str,
    review_mask: pd.Series,
) -> dict:
    pred_col = f"{model}_pred"

    review_mask = review_mask.fillna(False).astype(bool)

    true_high_risk = df["true_high_risk"]
    dangerous = df["high_risk_to_monitor"]

    total = len(df)

    review_count = int(review_mask.sum())

    dangerous_total = int(dangerous.sum())

    dangerous_caught = int(
        (dangerous & review_mask).sum()
    )

    dangerous_remaining = (
        dangerous_total - dangerous_caught
    )

    correct = (
        df[pred_col] == df["true_class"]
    )

    unnecessary_review = (
        review_mask & correct
    )

    incorrect_not_reviewed = (
        (~review_mask)
        & (~correct)
    )

    return {
        "model": model,
        "gate": gate_name,
        "N": total,
        "review_count": review_count,
        "review_rate": (
            review_count / total
            if total
            else np.nan
        ),
        "dangerous_high_risk_to_monitor": (
            dangerous_total
        ),
        "dangerous_caught": dangerous_caught,
        "dangerous_catch_rate": (
            dangerous_caught / dangerous_total
            if dangerous_total
            else np.nan
        ),
        "dangerous_remaining": dangerous_remaining,
        "correct_predictions_sent_to_review": int(
            unnecessary_review.sum()
        ),
        "correct_review_rate": (
            unnecessary_review.sum()
            / correct.sum()
            if correct.sum()
            else np.nan
        ),
        "incorrect_predictions_not_reviewed": int(
            incorrect_not_reviewed.sum()
        ),
    }


def global_confidence_gates(
    df: pd.DataFrame,
    model: str,
) -> list[dict]:
    confidence_col = f"{model}_confidence"

    rows = []

    for threshold in CONFIDENCE_THRESHOLDS:
        review = (
            df[confidence_col] < threshold
        )

        rows.append(
            evaluate_gate(
                df,
                model,
                f"global_confidence_lt_{threshold:.2f}",
                review,
            )
        )

    return rows


def low_risk_prediction_gate(
    df: pd.DataFrame,
    model: str,
) -> dict:
    review = df["predicted_low_risk"]

    return evaluate_gate(
        df,
        model,
        "all_low_risk_predictions_review",
        review,
    )


def low_risk_confidence_gates(
    df: pd.DataFrame,
    model: str,
) -> list[dict]:
    confidence_col = f"{model}_confidence"

    rows = []

    for threshold in CONFIDENCE_THRESHOLDS:
        review = (
            df["predicted_low_risk"]
            & (df[confidence_col] < threshold)
        )

        rows.append(
            evaluate_gate(
                df,
                model,
                (
                    "low_risk_and_confidence_lt_"
                    f"{threshold:.2f}"
                ),
                review,
            )
        )

    return rows


def high_risk_probability_gates(
    df: pd.DataFrame,
    model: str,
) -> list[dict]:
    if "predicted_high_risk_probability" not in df:
        return []

    rows = []

    for threshold in (
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ):
        review = (
            df["predicted_low_risk"]
            & (
                df["predicted_high_risk_probability"]
                >= threshold
            )
        )

        rows.append(
            evaluate_gate(
                df,
                model,
                (
                    "low_risk_and_high_risk_prob_ge_"
                    f"{threshold:.2f}"
                ),
                review,
            )
        )

    return rows


def print_model_summary(
    df: pd.DataFrame,
    model: str,
) -> None:
    pred_col = f"{model}_pred"

    print()
    print("=" * 80)
    print(f"{model.upper()} SAFETY BASELINE")
    print("=" * 80)

    total = len(df)

    dangerous = int(
        df["high_risk_to_monitor"].sum()
    )

    high_risk = int(
        df["true_high_risk"].sum()
    )

    correct = int(
        (df[pred_col] == df["true_class"]).sum()
    )

    print(f"Total cases:                    {total}")
    print(f"True high-risk cases:           {high_risk}")
    print(f"Correct predictions:            {correct}")
    print(
        f"High-risk → monitor failures:   "
        f"{dangerous}"
    )

    print()
    print("Dangerous failures:")
    print(
        df.loc[
            df["high_risk_to_monitor"],
            [
                "true_class",
                pred_col,
                "image_id",
                f"{model}_confidence",
            ],
        ].to_string(index=False)
    )


def print_results(
    results: pd.DataFrame,
    model: str,
) -> None:
    subset = results[
        results["model"] == model
    ].copy()

    print()
    print("=" * 80)
    print(f"{model.upper()} SAFETY GATE RESULTS")
    print("=" * 80)

    columns = [
        "gate",
        "review_rate",
        "dangerous_high_risk_to_monitor",
        "dangerous_caught",
        "dangerous_catch_rate",
        "dangerous_remaining",
        "correct_predictions_sent_to_review",
    ]

    display = subset[columns].copy()

    display["review_rate"] = (
        display["review_rate"]
        .map(pct)
    )

    display["dangerous_catch_rate"] = (
        display["dangerous_catch_rate"]
        .map(
            lambda x: (
                pct(x)
                if pd.notna(x)
                else "N/A"
            )
        )
    )

    print(
        display.to_string(index=False)
    )


def save_dangerous_cases(
    df: pd.DataFrame,
    model: str,
) -> None:
    pred_col = f"{model}_pred"
    confidence_col = f"{model}_confidence"

    cases = df[
        df["high_risk_to_monitor"]
    ].copy()

    columns = [
        "image_id",
        "true_class",
        pred_col,
        confidence_col,
    ]

    if "predicted_high_risk_probability" in cases:
        columns.append(
            "predicted_high_risk_probability"
        )

    cases[columns].to_csv(
        OUTPUT_DIR
        / f"{model}_dangerous_cases.csv",
        index=False,
    )


def main() -> None:
    print("=" * 80)
    print(
        "DERMASENSE PHASE 4 "
        "SAFETY GATE SIMULATION"
    )
    print("=" * 80)

    df = load_predictions()

    print(f"Input: {INPUT}")
    print(f"Rows:  {len(df)}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    prepared = {}

    for model in ("c1", "f1"):
        work = add_safety_columns(
            df,
            model,
        )

        work, probability_available = (
            add_high_risk_probability(
                work,
                model,
            )
        )

        prepared[model] = work

        print_model_summary(
            work,
            model,
        )

        all_results.extend(
            global_confidence_gates(
                work,
                model,
            )
        )

        all_results.append(
            low_risk_prediction_gate(
                work,
                model,
            )
        )

        all_results.extend(
            low_risk_confidence_gates(
                work,
                model,
            )
        )

        if probability_available:
            all_results.extend(
                high_risk_probability_gates(
                    work,
                    model,
                )
            )

        save_dangerous_cases(
            work,
            model,
        )

    results = pd.DataFrame(
        all_results
    )

    results.to_csv(
        OUTPUT_DIR / "safety_gate_results.csv",
        index=False,
    )

    for model in ("c1", "f1"):
        print_results(
            results,
            model,
        )

    print()
    print("=" * 80)
    print("MEL → NEV STRESS TEST")
    print("=" * 80)

    for model, work in prepared.items():
        case = work[
            (work["true_class"] == "MEL")
            & (work[f"{model}_pred"] == "NEV")
        ]

        if case.empty:
            print(
                f"{model.upper()}: "
                "MEL → NEV not found"
            )
            continue

        confidence = float(
            case.iloc[0][
                f"{model}_confidence"
            ]
        )

        print(
            f"{model.upper()}: "
            f"confidence={confidence:.4f}"
        )

        for _, row in case.iterrows():
            if (
                "predicted_high_risk_probability"
                in work.columns
            ):
                print(
                    "  predicted high-risk "
                    f"probability="
                    f"{row['predicted_high_risk_probability']:.4f}"
                )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)
    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print()
    print("=" * 80)
    print("PHASE 4 SAFETY GATE SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
