from pathlib import Path

import pandas as pd


INPUT = Path(
    "analysis/product_eval/c1_f1_test_predictions.csv"
)


HIGH_RISK = {"BCC", "MEL", "SCC"}

# Product-level action mapping.
ACTION = {
    "BCC": "URGENT_EVALUATION",
    "MEL": "URGENT_EVALUATION",
    "SCC": "URGENT_EVALUATION",
    "ACK": "EVALUATE_SOON",
    "NEV": "MONITOR",
    "SEK": "MONITOR",
}


def pct(x):
    return f"{100.0 * x:.1f}%"


def print_rate_table(df, model):
    pred_col = f"{model}_pred"

    print()
    print("=" * 80)
    print(f"{model.upper()} HIGH-RISK DOWNGRADE RATES")
    print("=" * 80)

    rows = []

    for source in ["BCC", "SCC", "MEL"]:
        source_df = df[df["true_class"] == source]

        n = len(source_df)

        downgraded = source_df[
            ~source_df[pred_col].isin(HIGH_RISK)
        ]

        downgrade_n = len(downgraded)

        rows.append(
            {
                "source": source,
                "total": n,
                "downgrade_n": downgrade_n,
                "downgrade_rate": (
                    downgrade_n / n
                    if n
                    else float("nan")
                ),
            }
        )

    table = pd.DataFrame(rows)

    for _, row in table.iterrows():
        print(
            f"{row['source']:>3}: "
            f"{int(row['downgrade_n'])}/"
            f"{int(row['total'])} = "
            f"{pct(row['downgrade_rate'])}"
        )

    return table


def bcc_ack_lesions(df, model):
    pred_col = f"{model}_pred"

    subset = df[
        (df["true_class"] == "BCC")
        & (df[pred_col] == "ACK")
    ]

    if "lesion_uid" in subset.columns:
        return set(subset["lesion_uid"].astype(str))

    if "patient_id" in subset.columns:
        return set(
            subset["patient_id"].astype(str)
            + "::"
            + subset["lesion_id"].astype(str)
        )

    return set(subset["image_id"].astype(str))


def print_bcc_ack_overlap(df):
    print()
    print("=" * 80)
    print("C1 vs F1 BCC → ACK LESION OVERLAP")
    print("=" * 80)

    c1 = bcc_ack_lesions(df, "c1")
    f1 = bcc_ack_lesions(df, "f1")

    both = c1 & f1
    c1_only = c1 - f1
    f1_only = f1 - c1

    print(f"C1 BCC → ACK lesions: {len(c1)}")
    print(f"F1 BCC → ACK lesions: {len(f1)}")
    print(f"Same in both models:  {len(both)}")
    print(f"C1-only:              {len(c1_only)}")
    print(f"F1-only:              {len(f1_only)}")

    print()
    print("Intersection:")
    for x in sorted(both):
        print(f"  {x}")

    if c1_only:
        print()
        print("C1-only:")
        for x in sorted(c1_only):
            print(f"  {x}")

    if f1_only:
        print()
        print("F1-only:")
        for x in sorted(f1_only):
            print(f"  {x}")


def classify_consequence(true_class, predicted_class):
    true_action = ACTION[true_class]
    predicted_action = ACTION[predicted_class]

    if (
        true_action == "URGENT_EVALUATION"
        and predicted_action == "MONITOR"
    ):
        return "HIGH_RISK_TO_MONITOR"

    if (
        true_action == "URGENT_EVALUATION"
        and predicted_action == "EVALUATE_SOON"
    ):
        return "HIGH_RISK_TO_EVALUATE_SOON"

    if (
        true_action == "URGENT_EVALUATION"
        and predicted_action == "URGENT_EVALUATION"
    ):
        return "HIGH_RISK_TO_HIGH_RISK"

    return "OTHER"


def print_action_consequences(df, model):
    pred_col = f"{model}_pred"

    work = df.copy()

    work["true_action"] = work["true_class"].map(ACTION)
    work["pred_action"] = work[pred_col].map(ACTION)

    work["safety_consequence"] = [
        classify_consequence(t, p)
        for t, p in zip(
            work["true_class"],
            work[pred_col],
        )
    ]

    print()
    print("=" * 80)
    print(f"{model.upper()} PRODUCT ACTION CONSEQUENCES")
    print("=" * 80)

    high_risk = work[
        work["true_class"].isin(HIGH_RISK)
    ]

    counts = (
        high_risk["safety_consequence"]
        .value_counts()
    )

    for category in [
        "HIGH_RISK_TO_MONITOR",
        "HIGH_RISK_TO_EVALUATE_SOON",
        "HIGH_RISK_TO_HIGH_RISK",
    ]:
        print(
            f"{category:<28}: "
            f"{int(counts.get(category, 0))}"
        )

    print()
    print("High-risk → monitor destinations:")

    monitor = high_risk[
        high_risk["safety_consequence"]
        == "HIGH_RISK_TO_MONITOR"
    ]

    if monitor.empty:
        print("  None")
    else:
        print(
            monitor[
                [
                    "true_class",
                    pred_col,
                    "image_id",
                ]
            ]
            .value_counts()
            .to_string()
        )

    return work


def print_destination_confidence(df, model):
    pred_col = f"{model}_pred"
    conf_col = f"{model}_confidence"

    if conf_col not in df.columns:
        raise RuntimeError(
            f"Missing confidence column: {conf_col}"
        )

    work = df.copy()

    work["safety_consequence"] = [
        classify_consequence(t, p)
        for t, p in zip(
            work["true_class"],
            work[pred_col],
        )
    ]

    high_risk_errors = work[
        work["true_class"].isin(HIGH_RISK)
        & (
            work["safety_consequence"]
            .isin(
                [
                    "HIGH_RISK_TO_MONITOR",
                    "HIGH_RISK_TO_EVALUATE_SOON",
                ]
            )
        )
    ].copy()

    print()
    print("=" * 80)
    print(f"{model.upper()} CONFIDENCE BY SAFETY DESTINATION")
    print("=" * 80)

    for consequence in [
        "HIGH_RISK_TO_MONITOR",
        "HIGH_RISK_TO_EVALUATE_SOON",
    ]:
        subset = high_risk_errors[
            high_risk_errors["safety_consequence"]
            == consequence
        ]

        print()
        print(consequence)
        print("-" * 80)

        if subset.empty:
            print("N = 0")
            continue

        grouped = (
            subset
            .groupby(
                ["true_class", pred_col],
                dropna=False,
            )[conf_col]
            .agg(
                N="count",
                mean="mean",
                median="median",
                minimum="min",
                maximum="max",
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            values = subset[
                (subset["true_class"] == row["true_class"])
                & (subset[pred_col] == row[pred_col])
            ][conf_col]

            high_conf = (
                (values >= 0.90).mean()
            )

            print(
                f"{row['true_class']} → "
                f"{row[pred_col]} | "
                f"N={int(row['N']):2d} | "
                f"mean={row['mean']:.4f} | "
                f"median={row['median']:.4f} | "
                f"min={row['minimum']:.4f} | "
                f"max={row['maximum']:.4f} | "
                f">=0.90={pct(high_conf)}"
            )


def main():
    print("=" * 80)
    print("DERMASENSE PHASE 4 SAFETY CONSEQUENCE AUDIT")
    print("=" * 80)

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    print(f"Input: {INPUT}")
    print(f"Rows:  {len(df)}")

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

    c1_rates = print_rate_table(df, "c1")
    f1_rates = print_rate_table(df, "f1")

    print_bcc_ack_overlap(df)

    c1_actions = print_action_consequences(
        df,
        "c1",
    )

    f1_actions = print_action_consequences(
        df,
        "f1",
    )

    print_destination_confidence(df, "c1")
    print_destination_confidence(df, "f1")

    output_dir = Path(
        "analysis/product_eval/"
        "phase4_safety_consequence"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    c1_rates.to_csv(
        output_dir / "c1_high_risk_downgrade_rates.csv",
        index=False,
    )

    f1_rates.to_csv(
        output_dir / "f1_high_risk_downgrade_rates.csv",
        index=False,
    )

    c1_actions.to_csv(
        output_dir / "c1_action_consequences.csv",
        index=False,
    )

    f1_actions.to_csv(
        output_dir / "f1_action_consequences.csv",
        index=False,
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
