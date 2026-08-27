"""
DermaSense CV-1 resolution/detail threshold calibration.

This script calibrates the effective-detail reference used by the
CV-1 resolution signal.

Important:
    - Original images are never modified.
    - Degraded images exist only in memory.
    - This script does NOT modify src/quality.
    - This is an engineering calibration experiment, not clinical
      validation.

For each candidate detail reference, the script measures:

    - clean-image rejection
    - resolution degradation detection at severity 1
    - resolution degradation detection at severity 2
    - resolution degradation detection at severity 3

The result is used to select an operating point before changing
the production CV-1 implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.validate_cv1_robustness import (
    DEFAULT_SAMPLE_SIZE,
    SEED,
    reduce_resolution,
)


OUTPUT_DIR = Path(
    "analysis/quality/cv1_calibration"
)

DEFAULT_REFERENCES = (
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    50.0,
    60.0,
    75.0,
    100.0,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the CV-1 effective-detail "
            "resolution reference."
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


def effective_detail(image: np.ndarray) -> float:
    """
    Measure effective high-frequency spatial detail.

    The raw Laplacian variance is intentionally retained as the
    calibration quantity.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


def resolution_score(
    detail: float,
    reference: float,
) -> float:
    """
    Convert effective detail into a normalized resolution score.
    """

    return float(
        np.clip(
            detail / reference,
            0.0,
            1.0,
        )
    )


def sample_manifest(
    manifest_path: str,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    required_columns = {
        "dataset",
        "image_id",
        "image_path",
        "native_diagnosis",
        "split",
    }

    missing = required_columns.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    test_df = df[
        df["split"] == "test"
    ].copy()

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be positive"
        )

    if sample_size > len(test_df):
        raise ValueError(
            f"Requested {sample_size} samples, "
            f"but only {len(test_df)} test images exist."
        )

    # Stratify across native diagnosis where possible.
    sampled_parts = []

    classes = sorted(
        test_df["native_diagnosis"]
        .dropna()
        .unique()
    )

    per_class = sample_size // len(classes)
    remainder = sample_size % len(classes)

    rng = np.random.default_rng(seed)

    for index, class_name in enumerate(classes):
        class_df = test_df[
            test_df["native_diagnosis"]
            == class_name
        ]

        n = per_class + (
            1 if index < remainder else 0
        )

        if n > len(class_df):
            raise ValueError(
                f"Not enough samples for class "
                f"{class_name!r}."
            )

        indices = rng.choice(
            len(class_df),
            size=n,
            replace=False,
        )

        sampled_parts.append(
            class_df.iloc[
                indices
            ]
        )

    sampled = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    return sampled.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {path}"
        )

    return image


def collect_detail_measurements(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        image = load_image(
            row["image_path"]
        )

        clean_detail = effective_detail(
            image
        )

        rows.append(
            {
                "image_id": row["image_id"],
                "native_diagnosis": row[
                    "native_diagnosis"
                ],
                "clean_detail": clean_detail,
            }
        )

        for severity in (
            1,
            2,
            3,
        ):
            degraded = reduce_resolution(
                image,
                severity,
            )

            degraded_detail = effective_detail(
                degraded
            )

            rows.append(
                {
                    "image_id": row["image_id"],
                    "native_diagnosis": row[
                        "native_diagnosis"
                    ],
                    "clean_detail": clean_detail,
                    "severity": severity,
                    "degraded_detail": (
                        degraded_detail
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_case_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        image = load_image(
            row["image_path"]
        )

        clean_detail = effective_detail(
            image
        )

        rows.append(
            {
                "image_id": row["image_id"],
                "native_diagnosis": row[
                    "native_diagnosis"
                ],
                "degradation": "clean",
                "severity": 0,
                "detail": clean_detail,
            }
        )

        for severity in (
            1,
            2,
            3,
        ):
            degraded = reduce_resolution(
                image,
                severity,
            )

            rows.append(
                {
                    "image_id": row["image_id"],
                    "native_diagnosis": row[
                        "native_diagnosis"
                    ],
                    "degradation": "resolution",
                    "severity": severity,
                    "detail": effective_detail(
                        degraded
                    ),
                }
            )

    return pd.DataFrame(rows)


def evaluate_reference(
    cases: pd.DataFrame,
    reference: float,
) -> dict[str, float]:
    clean = cases[
        cases["degradation"] == "clean"
    ]

    resolution = cases[
        cases["degradation"]
        == "resolution"
    ]

    clean_scores = np.clip(
        clean["detail"].to_numpy()
        / reference,
        0.0,
        1.0,
    )

    clean_reject = float(
        np.mean(clean_scores < 0.50)
    )

    result = {
        "reference": reference,
        "clean_reject": clean_reject,
    }

    for severity in (
        1,
        2,
        3,
    ):
        subset = resolution[
            resolution["severity"]
            == severity
        ]

        scores = np.clip(
            subset["detail"].to_numpy()
            / reference,
            0.0,
            1.0,
        )

        result[
            f"resolution_s{severity}_catch"
        ] = float(
            np.mean(scores < 0.50)
        )

    return result


def main():
    args = parse_args()

    print("=" * 92)
    print(
        "DERMASENSE CV-1 RESOLUTION DETAIL "
        "THRESHOLD CALIBRATION"
    )
    print("=" * 92)

    print(
        f"Manifest:     {args.manifest}"
    )

    print(
        f"Sample size:  {args.sample_size}"
    )

    print(
        f"Seed:         {args.seed}"
    )

    references = DEFAULT_REFERENCES

    print(
        "References:   "
        + ", ".join(
            f"{value:.1f}"
            for value in references
        )
    )

    print()
    print(
        "Original images will not be modified."
    )

    df = sample_manifest(
        args.manifest,
        args.sample_size,
        args.seed,
    )

    print()
    print("Class distribution:")

    print(
        df["native_diagnosis"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("=" * 92)
    print(
        "EFFECTIVE DETAIL DISTRIBUTION"
    )
    print("=" * 92)

    cases = build_case_table(
        df
    )

    clean = cases[
        cases["degradation"] == "clean"
    ]

    resolution = cases[
        cases["degradation"]
        == "resolution"
    ]

    print(
        clean["detail"]
        .describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
            ]
        )
        .to_string()
    )

    print()
    print(
        "Resolution degradation detail:"
    )

    detail_summary = (
        resolution
        .groupby("severity")["detail"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(
        detail_summary.to_string(
            float_format=lambda value:
            f"{value:.3f}"
        )
    )

    print()
    print("=" * 92)
    print(
        "DETAIL REFERENCE ANALYSIS"
    )
    print("=" * 92)

    results = []

    for reference in references:
        results.append(
            evaluate_reference(
                cases,
                reference,
            )
        )

    result_df = pd.DataFrame(
        results
    )

    display_df = result_df.copy()

    percentage_columns = [
        "clean_reject",
        "resolution_s1_catch",
        "resolution_s2_catch",
        "resolution_s3_catch",
    ]

    for column in percentage_columns:
        display_df[column] = (
            display_df[column] * 100.0
        )

    print(
        display_df.to_string(
            index=False,
            formatters={
                "reference": (
                    lambda value:
                    f"{value:.1f}"
                ),
                "clean_reject": (
                    lambda value:
                    f"{value:.1f}%"
                ),
                "resolution_s1_catch": (
                    lambda value:
                    f"{value:.1f}%"
                ),
                "resolution_s2_catch": (
                    lambda value:
                    f"{value:.1f}%"
                ),
                "resolution_s3_catch": (
                    lambda value:
                    f"{value:.1f}%"
                ),
            },
        )
    )

    print()
    print("=" * 92)
    print("INTERPRETATION")
    print("=" * 92)

    print(
        """
This experiment does NOT modify src/quality.

Use the table to select a detail reference that:

  - keeps clean-image rejection reasonably low,
  - catches moderate resolution degradation,
  - catches severe resolution degradation,
  - preserves monotonic behavior,
  - avoids making CV-1 unnecessarily strict.

The selected reference should be justified from these
measurements before changing the production implementation.
"""
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        OUTPUT_DIR
        / "resolution_threshold_calibration.csv"
    )

    cases_path = (
        OUTPUT_DIR
        / "resolution_detail_cases.csv"
    )

    result_df.to_csv(
        result_path,
        index=False,
    )

    cases.to_csv(
        cases_path,
        index=False,
    )

    print(
        f"Saved: {result_path}"
    )

    print(
        f"Saved: {cases_path}"
    )

    print()
    print("=" * 92)
    print(
        "CV-1 RESOLUTION CALIBRATION COMPLETE"
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
