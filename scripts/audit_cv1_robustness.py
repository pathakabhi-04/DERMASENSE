"""
DermaSense CV-1 robustness audit.

Audits the already-generated controlled-degradation experiment.

This script does NOT:
    - regenerate images
    - modify CV-1 thresholds
    - modify source images
    - train a model

It answers four product-quality questions:

1. Why do clean images fail CV-1?
2. Does each degradation primarily affect its intended signal?
3. Does issue detection become stronger with degradation?
4. Does detected issue -> guidance remain context-correct?

The goal is diagnosis of the current CV-1 behavior, not optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS = Path(
    "analysis/quality/cv1_robustness/cv1_robustness_results.csv"
)

OUTPUT_DIR = Path(
    "analysis/quality/cv1_robustness/audit"
)


EXPECTED_GUIDANCE = {
    "resolution": (
        "Use a higher-resolution image and avoid excessive cropping."
    ),
    "low_brightness": (
        "Move to a well-lit area and retake the image without glare."
    ),
    "high_brightness": (
        "Avoid direct glare or harsh light and retake the image."
    ),
    "low_contrast": (
        "Improve the lighting and retake the image with the lesion clearly visible."
    ),
    "motion_blur": (
        "Keep the camera steady and retake the image."
    ),
}


DEGRADATION_TO_SIGNAL = {
    "brightness": "brightness",
    "contrast": "contrast",
    "blur": "blur",
    "resolution": "resolution",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit existing CV-1 robustness results."
    )

    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS),
    )

    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Robustness results do not exist: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "dataset",
        "image_id",
        "lesion_uid",
        "native_diagnosis",
        "degradation",
        "severity",
        "quality_score",
        "usable",
        "issues",
        "issue_severities",
        "recommended_action",
        "guidance",
    }

    missing = required.difference(df.columns)

    if missing:
        raise RuntimeError(
            "Robustness results are missing required columns: "
            f"{sorted(missing)}"
        )

    return df


def parse_issues(value) -> list[str]:
    if pd.isna(value):
        return []

    parsed = json.loads(value)

    if not isinstance(parsed, list):
        raise RuntimeError(
            f"Expected issue list, got: {type(parsed).__name__}"
        )

    return [
        str(item)
        for item in parsed
    ]


def parse_issue_severities(value) -> dict[str, float]:
    if pd.isna(value):
        return {}

    parsed = json.loads(value)

    if not isinstance(parsed, dict):
        raise RuntimeError(
            "Expected issue-severity dictionary."
        )

    return {
        str(key): float(val)
        for key, val in parsed.items()
    }


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: (
            value
            if isinstance(value, bool)
            else str(value).strip().lower()
            in {"true", "1", "yes"}
        )
    )


def audit_clean_failures(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = df[
        df["degradation"] == "clean"
    ].copy()

    clean["issues_parsed"] = clean[
        "issues"
    ].map(parse_issues)

    rows = []

    for _, row in clean.iterrows():
        issues = row["issues_parsed"]

        rows.append(
            {
                "image_id": row["image_id"],
                "lesion_uid": row["lesion_uid"],
                "native_diagnosis": row[
                    "native_diagnosis"
                ],
                "quality_score": row[
                    "quality_score"
                ],
                "usable": row["usable"],
                "issue_count": len(issues),
                "issues": "|".join(issues),
            }
        )

    clean_cases = pd.DataFrame(rows)

    failure_summary = (
        clean_cases[
            ~clean_cases["usable"]
        ]
        .assign(
            issue=lambda frame: frame[
                "issues"
            ].str.split("|")
        )
        .explode("issue")
        .query("issue.notna() and issue != ''")
        .groupby("issue")
        .size()
        .reset_index(name="failed_clean_images")
        .sort_values(
            "failed_clean_images",
            ascending=False,
        )
    )

    return (
        clean_cases,
        failure_summary,
    )


def audit_signal_specificity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for degradation, expected_signal in (
        DEGRADATION_TO_SIGNAL.items()
    ):
        subset = df[
            df["degradation"] == degradation
        ].copy()

        for severity in sorted(
            subset["severity"].unique()
        ):
            level = subset[
                subset["severity"] == severity
            ]

            signal_columns = [
                "resolution",
                "brightness",
                "contrast",
                "blur",
            ]

            for column in signal_columns:
                values = []

                for value in level[
                    "signals"
                ]:
                    parsed = json.loads(value)
                    values.append(
                        float(parsed[column])
                    )

                rows.append(
                    {
                        "degradation": degradation,
                        "severity": severity,
                        "signal": column,
                        "expected_signal": (
                            column
                            == expected_signal
                        ),
                        "mean_signal": sum(values)
                        / len(values),
                        "median_signal": (
                            pd.Series(values)
                            .median()
                        ),
                    }
                )

    return pd.DataFrame(rows)


def audit_issue_detection(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for degradation in DEGRADATION_TO_SIGNAL:
        subset = df[
            df["degradation"] == degradation
        ]

        expected_signal = (
            DEGRADATION_TO_SIGNAL[
                degradation
            ]
        )

        expected_issue = {
            "brightness": "low_brightness",
            "contrast": "low_contrast",
            "blur": "motion_blur",
            "resolution": "resolution",
        }[expected_signal]

        for severity in sorted(
            subset["severity"].unique()
        ):
            level = subset[
                subset["severity"] == severity
            ]

            issue_lists = level[
                "issues"
            ].map(parse_issues)

            expected_detected = issue_lists.map(
                lambda issues:
                expected_issue in issues
            )

            rows.append(
                {
                    "degradation": degradation,
                    "severity": severity,
                    "samples": len(level),
                    "expected_issue": expected_issue,
                    "expected_issue_detection_rate": (
                        expected_detected.mean()
                        * 100.0
                    ),
                    "mean_issue_count": (
                        issue_lists.map(len).mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def audit_guidance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        issues = parse_issues(
            row["issues"]
        )

        severities = parse_issue_severities(
            row["issue_severities"]
        )

        guidance = json.loads(
            row["guidance"]
        )

        guidance_by_issue = {
            item["type"]: item["guidance"]
            for item in guidance
            if isinstance(item, dict)
        }

        for issue in issues:
            expected = EXPECTED_GUIDANCE.get(
                issue
            )

            actual = guidance_by_issue.get(
                issue
            )

            rows.append(
                {
                    "image_id": row["image_id"],
                    "degradation": row[
                        "degradation"
                    ],
                    "severity": row["severity"],
                    "issue": issue,
                    "issue_severity": severities.get(
                        issue
                    ),
                    "guidance": actual,
                    "expected_guidance": expected,
                    "guidance_matches": (
                        actual == expected
                        if expected is not None
                        else False
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "image_id",
                "degradation",
                "severity",
                "issue",
                "issue_severity",
                "guidance",
                "expected_guidance",
                "guidance_matches",
            ]
        )

    return pd.DataFrame(rows)


def audit_monotonicity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for degradation in DEGRADATION_TO_SIGNAL:
        subset = df[
            df["degradation"] == degradation
        ].copy()

        expected_signal = (
            DEGRADATION_TO_SIGNAL[
                degradation
            ]
        )

        means = (
            subset
            .groupby("severity")[
                "signals"
            ]
            .apply(
                lambda series:
                pd.Series(
                    [
                        json.loads(value)[
                            expected_signal
                        ]
                        for value in series
                    ]
                ).mean()
            )
            .to_dict()
        )

        scores = (
            subset
            .groupby("severity")[
                "quality_score"
            ]
            .mean()
            .to_dict()
        )

        for severity in sorted(means):
            rows.append(
                {
                    "degradation": degradation,
                    "severity": severity,
                    "mean_target_signal": means[
                        severity
                    ],
                    "mean_quality_score": scores[
                        severity
                    ],
                }
            )

    return pd.DataFrame(rows)


def print_section(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main():
    args = parse_args()

    results_path = Path(
        args.results
    )

    print("=" * 80)
    print(
        "DERMASENSE CV-1 ROBUSTNESS AUDIT"
    )
    print("=" * 80)

    print(
        f"Results: {results_path}"
    )

    df = load_results(
        results_path
    )

    df["usable"] = normalize_bool(
        df["usable"]
    )

    print(
        f"Rows: {len(df)}"
    )

    # ------------------------------------------------------------------
    # 1. CLEAN FAILURE ANALYSIS
    # ------------------------------------------------------------------

    clean_cases, clean_failures = (
        audit_clean_failures(df)
    )

    print_section(
        "1. CLEAN IMAGE FAILURE ANALYSIS"
    )

    print(
        f"Clean images: {len(clean_cases)}"
    )

    print(
        f"Clean usable: "
        f"{clean_cases['usable'].sum()} "
        f"({clean_cases['usable'].mean() * 100:.1f}%)"
    )

    print(
        f"Clean failed: "
        f"{(~clean_cases['usable']).sum()} "
        f"({(~clean_cases['usable']).mean() * 100:.1f}%)"
    )

    print()
    print(
        "Failure issue distribution:"
    )

    if clean_failures.empty:
        print("No clean-image failures.")
    else:
        print(
            clean_failures.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # 2. SIGNAL SPECIFICITY
    # ------------------------------------------------------------------

    signal_audit = audit_signal_specificity(
        df
    )

    print_section(
        "2. SIGNAL-SPECIFIC DEGRADATION RESPONSE"
    )

    print(
        signal_audit.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # 3. ISSUE DETECTION
    # ------------------------------------------------------------------

    detection_audit = audit_issue_detection(
        df
    )

    print_section(
        "3. EXPECTED ISSUE DETECTION"
    )

    print(
        detection_audit.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # 4. GUIDANCE
    # ------------------------------------------------------------------

    guidance_audit = audit_guidance(
        df
    )

    print_section(
        "4. CONTEXT-AWARE GUIDANCE VALIDATION"
    )

    if guidance_audit.empty:
        print(
            "No issues were detected."
        )
    else:
        total_guidance = len(
            guidance_audit
        )

        correct_guidance = int(
            guidance_audit[
                "guidance_matches"
            ].sum()
        )

        print(
            f"Guidance checks: {total_guidance}"
        )

        print(
            f"Correct mappings: "
            f"{correct_guidance}"
        )

        print(
            f"Guidance correctness: "
            f"{correct_guidance / total_guidance * 100:.1f}%"
        )

        mismatches = guidance_audit[
            ~guidance_audit[
                "guidance_matches"
            ]
        ]

        if not mismatches.empty:
            print()
            print(
                "Guidance mismatches:"
            )

            print(
                mismatches.to_string(
                    index=False
                )
            )

    # ------------------------------------------------------------------
    # 5. MONOTONICITY
    # ------------------------------------------------------------------

    monotonicity = audit_monotonicity(
        df
    )

    print_section(
        "5. DEGRADATION MONOTONICITY"
    )

    print(
        monotonicity.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_cases.to_csv(
        OUTPUT_DIR
        / "clean_image_cases.csv",
        index=False,
    )

    clean_failures.to_csv(
        OUTPUT_DIR
        / "clean_failure_summary.csv",
        index=False,
    )

    signal_audit.to_csv(
        OUTPUT_DIR
        / "signal_specificity.csv",
        index=False,
    )

    detection_audit.to_csv(
        OUTPUT_DIR
        / "issue_detection.csv",
        index=False,
    )

    guidance_audit.to_csv(
        OUTPUT_DIR
        / "guidance_validation.csv",
        index=False,
    )

    monotonicity.to_csv(
        OUTPUT_DIR
        / "monotonicity.csv",
        index=False,
    )

    print_section(
        "AUDIT COMPLETE"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
