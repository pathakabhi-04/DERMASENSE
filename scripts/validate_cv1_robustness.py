"""
DermaSense CV-1 controlled degradation robustness validation.

This experiment evaluates whether the image-quality module responds
sensibly to controlled image degradation.

Important:
    - Original test images are never modified.
    - Degraded images exist only in memory.
    - This is an engineering robustness test, not clinical validation.
    - CV-1 thresholds are NOT changed by this script.

Experiment:
    clean
      -> brightness degradation
      -> contrast degradation
      -> blur degradation
      -> resolution degradation
      -> selected combined degradation

For each image/degradation condition, the script records:
    - quality score
    - usability
    - detected issues
    - issue severities
    - recommended action
    - context-aware guidance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.quality.assessment import assess_image


SEED = 42

DEFAULT_SAMPLE_SIZE = 48

OUTPUT_DIR = Path(
    "analysis/quality/cv1_robustness"
)

DEGRADATIONS = (
    ("clean", 0),
    ("brightness", 1),
    ("brightness", 2),
    ("brightness", 3),
    ("contrast", 1),
    ("contrast", 2),
    ("contrast", 3),
    ("blur", 1),
    ("blur", 2),
    ("blur", 3),
    ("resolution", 1),
    ("resolution", 2),
    ("resolution", 3),
    ("combined", 1),
    ("combined", 2),
    ("combined", 3),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate CV-1 image-quality robustness "
            "under controlled image degradation."
        )
    )

    parser.add_argument(
        "--manifest",
        default="data/splits/pad_ufes/test.csv",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    return parser.parse_args()


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Test split does not exist: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "dataset",
        "image_id",
        "lesion_uid",
        "image_path",
        "native_diagnosis",
        "split",
    }

    missing = required.difference(df.columns)

    if missing:
        raise RuntimeError(
            "Manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    if not (df["split"] == "test").all():
        raise RuntimeError(
            "Expected all rows in the supplied manifest "
            "to belong to the test split."
        )

    return df


def stratified_sample(
    df: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError(
            "sample_size must be positive."
        )

    if sample_size >= len(df):
        return df.sample(
            frac=1.0,
            random_state=seed,
        ).reset_index(drop=True)

    rng = np.random.default_rng(seed)

    classes = sorted(
        df["native_diagnosis"].unique()
    )

    selected_indices = []

    base_n = sample_size // len(classes)
    remainder = sample_size % len(classes)

    for index, class_name in enumerate(classes):
        class_df = df[
            df["native_diagnosis"] == class_name
        ]

        n = base_n + (
            1 if index < remainder else 0
        )

        n = min(n, len(class_df))

        if n > 0:
            chosen = rng.choice(
                class_df.index.to_numpy(),
                size=n,
                replace=False,
            )
            selected_indices.extend(
                chosen.tolist()
            )

    sampled = df.loc[
        selected_indices
    ].copy()

    # If a very small class prevented exact allocation,
    # fill the remainder from the unused pool.
    if len(sampled) < sample_size:
        remaining = df.drop(
            index=sampled.index
        )

        extra_n = min(
            sample_size - len(sampled),
            len(remaining),
        )

        if extra_n > 0:
            extra = remaining.sample(
                n=extra_n,
                random_state=seed,
            )

            sampled = pd.concat(
                [sampled, extra]
            )

    return sampled.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {path}"
        )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Could not decode image: {path}"
        )

    return image


def adjust_brightness(
    image: np.ndarray,
    severity: int,
) -> np.ndarray:
    factors = {
        1: 0.75,
        2: 0.50,
        3: 0.30,
    }

    factor = factors[severity]

    output = image.astype(
        np.float32
    ) * factor

    return np.clip(
        output,
        0,
        255,
    ).astype(np.uint8)


def adjust_contrast(
    image: np.ndarray,
    severity: int,
) -> np.ndarray:
    factors = {
        1: 0.70,
        2: 0.45,
        3: 0.25,
    }

    factor = factors[severity]

    output = (
        image.astype(np.float32) - 127.5
    ) * factor + 127.5

    return np.clip(
        output,
        0,
        255,
    ).astype(np.uint8)


def apply_blur(
    image: np.ndarray,
    severity: int,
) -> np.ndarray:
    kernels = {
        1: 5,
        2: 11,
        3: 21,
    }

    kernel = kernels[severity]

    return cv2.GaussianBlur(
        image,
        (kernel, kernel),
        0,
    )


def reduce_resolution(
    image: np.ndarray,
    severity: int,
) -> np.ndarray:
    scale = {
        1: 0.50,
        2: 0.25,
        3: 0.125,
    }[severity]

    height, width = image.shape[:2]

    reduced_width = max(
        8,
        int(width * scale),
    )

    reduced_height = max(
        8,
        int(height * scale),
    )

    reduced = cv2.resize(
        image,
        (reduced_width, reduced_height),
        interpolation=cv2.INTER_AREA,
    )

    return cv2.resize(
        reduced,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def apply_degradation(
    image: np.ndarray,
    degradation: str,
    severity: int,
) -> np.ndarray:
    if degradation == "clean":
        return image.copy()

    if degradation == "brightness":
        return adjust_brightness(
            image,
            severity,
        )

    if degradation == "contrast":
        return adjust_contrast(
            image,
            severity,
        )

    if degradation == "blur":
        return apply_blur(
            image,
            severity,
        )

    if degradation == "resolution":
        return reduce_resolution(
            image,
            severity,
        )

    if degradation == "combined":
        output = adjust_brightness(
            image,
            severity,
        )

        output = adjust_contrast(
            output,
            severity,
        )

        output = apply_blur(
            output,
            severity,
        )

        output = reduce_resolution(
            output,
            severity,
        )

        return output

    raise ValueError(
        f"Unknown degradation: {degradation}"
    )


def result_to_record(
    *,
    metadata: pd.Series,
    degradation: str,
    severity: int,
    result,
) -> dict:
    issues = getattr(
        result,
        "issues",
        [],
    )

    issue_types = []
    issue_severities = {}

    for issue in issues:
        if isinstance(issue, dict):
            issue_type = issue.get(
                "type",
                "unknown",
            )

            issue_severity = issue.get(
                "severity",
                None,
            )

        else:
            issue_type = getattr(
                issue,
                "type",
                "unknown",
            )

            issue_severity = getattr(
                issue,
                "severity",
                None,
            )

        issue_types.append(
            str(issue_type)
        )

        if issue_severity is not None:
            issue_severities[
                str(issue_type)
            ] = float(issue_severity)

    return {
        "dataset": metadata["dataset"],
        "image_id": metadata["image_id"],
        "lesion_uid": metadata["lesion_uid"],
        "native_diagnosis": metadata[
            "native_diagnosis"
        ],
        "degradation": degradation,
        "severity": severity,
        "quality_score": float(
            result.quality_score
        ),
        "usable": bool(
            result.usable
        ),
        "issues": json.dumps(
            issue_types
        ),
        "signals": json.dumps(
            result.signals, sort_keys=True
        ),
        "issue_severities": json.dumps(
            issue_severities
        ),
        "recommended_action": str(
            result.recommended_action
        ),
        "guidance": json.dumps(
            [
                {
                    "type": issue.type,
                    "severity": float(issue.severity),
                    "guidance": issue.guidance,
                }
                for issue in result.issues
            ]
        ),
    }


def run_experiment(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    total = len(manifest)

    for image_index, (_, metadata) in enumerate(
        manifest.iterrows(),
        start=1,
    ):
        image_path = Path(
            metadata["image_path"]
        )

        image = load_image(
            image_path
        )

        print(
            f"[{image_index:>3}/{total}] "
            f"{metadata['image_id']}"
        )

        for degradation, severity in DEGRADATIONS:
            degraded = apply_degradation(
                image,
                degradation,
                severity,
            )

            result = assess_image(
                degraded
            )

            records.append(
                result_to_record(
                    metadata=metadata,
                    degradation=degradation,
                    severity=severity,
                    result=result,
                )
            )

    return pd.DataFrame(records)


def summarize(results: pd.DataFrame):
    summary = (
        results
        .groupby(
            ["degradation", "severity"],
            dropna=False,
        )
        .agg(
            samples=(
                "quality_score",
                "count",
            ),
            mean_quality_score=(
                "quality_score",
                "mean",
            ),
            median_quality_score=(
                "quality_score",
                "median",
            ),
            usable_rate=(
                "usable",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["usable_rate"] *= 100.0

    return summary


def issue_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in results.iterrows():
        issues = json.loads(
            row["issues"]
        )

        for issue_type in issues:
            rows.append(
                {
                    "degradation": row[
                        "degradation"
                    ],
                    "severity": row[
                        "severity"
                    ],
                    "issue_type": issue_type,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "degradation",
                "severity",
                "issue_type",
                "count",
            ]
        )

    return (
        pd.DataFrame(rows)
        .groupby(
            [
                "degradation",
                "severity",
                "issue_type",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            [
                "degradation",
                "severity",
                "count",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
    )


def write_summary(
    summary: pd.DataFrame,
    issues: pd.DataFrame,
):
    summary_path = (
        OUTPUT_DIR
        / "cv1_robustness_summary.csv"
    )

    issue_path = (
        OUTPUT_DIR
        / "cv1_robustness_issue_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    issues.to_csv(
        issue_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("CV-1 ROBUSTNESS SUMMARY")
    print("=" * 80)

    display_summary = summary.copy()

    display_summary[
        "mean_quality_score"
    ] = display_summary[
        "mean_quality_score"
    ].round(4)

    display_summary[
        "median_quality_score"
    ] = display_summary[
        "median_quality_score"
    ].round(4)

    display_summary[
        "usable_rate"
    ] = display_summary[
        "usable_rate"
    ].round(1)

    print(
        display_summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {summary_path}"
    )

    print(
        f"Saved: {issue_path}"
    )


def main():
    args = parse_args()

    manifest_path = Path(
        args.manifest
    )

    print("=" * 80)
    print(
        "DERMASENSE CV-1 CONTROLLED "
        "DEGRADATION ROBUSTNESS"
    )
    print("=" * 80)

    print(
        f"Manifest:     {manifest_path}"
    )

    print(
        f"Sample size:  {args.sample_size}"
    )

    print(
        f"Seed:         {args.seed}"
    )

    print()

    manifest = load_manifest(
        manifest_path
    )

    sample = stratified_sample(
        manifest,
        args.sample_size,
        args.seed,
    )

    print(
        "Class distribution:"
    )

    print(
        sample[
            "native_diagnosis"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Original images will not be modified."
    )

    results = run_experiment(
        sample
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        OUTPUT_DIR
        / "cv1_robustness_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    summary = summarize(
        results
    )

    issues = issue_summary(
        results
    )

    write_summary(
        summary,
        issues,
    )

    print()
    print("=" * 80)
    print("CV-1 ROBUSTNESS TEST COMPLETE")
    print("=" * 80)

    print(
        f"Detailed results: {results_path}"
    )


if __name__ == "__main__":
    main()
