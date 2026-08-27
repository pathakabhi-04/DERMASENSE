"""
Analyze SCC test-lesion coverage relative to the C1 training feature space.

Question:
Are problematic SCC test lesions systematically farther from the
training SCC representation than clean SCC test lesions?

Uses:
- C1 seed-42 backbone features
- PAD-UFES train/test split
- Existing SCC -> BCC error analysis
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu


FEATURE_DIR = Path("analysis/scc_bcc")

PAD_FEATURES = (
    FEATURE_DIR
    / "pad_ufes_test_scc_bcc_features.npz"
)

# We need training features too. These will be generated if absent.
TRAIN_FEATURES = (
    FEATURE_DIR
    / "pad_ufes_train_scc_bcc_features.npz"
)

MANIFEST = Path(
    "data/manifests/pad_ufes_manifest.csv"
)

TRAIN_SPLIT = Path(
    "data/splits/pad_ufes/train.csv"
)

TEST_SPLIT = Path(
    "data/splits/pad_ufes/test.csv"
)

ERROR_CSV = (
    FEATURE_DIR
    / "c1_seed42_errors"
    / "scc_to_bcc_errors.csv"
)

OUTPUT_DIR = (
    FEATURE_DIR
    / "training_coverage"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def make_lesion_uid(df: pd.DataFrame) -> pd.Series:
    return (
        df["patient_id"].astype(str)
        + "__"
        + df["lesion_id"].astype(str)
    )


def load_feature_file(path: Path):
    data = np.load(path, allow_pickle=True)

    print(f"Loading: {path}")

    for key in data.files:
        value = data[key]
        print(
            f"  {key}: "
            f"shape={getattr(value, 'shape', None)}"
        )

    return data


def main():
    print("=" * 80)
    print("DERMASENSE SCC TRAINING-COVERAGE ANALYSIS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    manifest = pd.read_csv(MANIFEST)
    train = pd.read_csv(TRAIN_SPLIT)
    test = pd.read_csv(TEST_SPLIT)
    errors = pd.read_csv(ERROR_CSV)

    manifest["lesion_uid"] = make_lesion_uid(
        manifest
    )
    train["lesion_uid"] = make_lesion_uid(train)
    test["lesion_uid"] = make_lesion_uid(test)

    error_ids = set(
        errors["image_id"].astype(str)
    )

    print()
    print("DATA")
    print(f"Manifest: {len(manifest)}")
    print(f"Train:    {len(train)}")
    print(f"Test:     {len(test)}")
    print(f"Error images: {len(error_ids)}")

    # ------------------------------------------------------------------
    # Load existing test features
    # ------------------------------------------------------------------

    test_features = load_feature_file(
        PAD_FEATURES
    )

    test_x = test_features["features"]

    # The existing feature-analysis NPZ stores only
    # features and targets. Feature extraction uses
    # shuffle=False, so rows correspond exactly to
    # the dataset split ordering.
    if len(test_x) != len(test):
        raise RuntimeError(
            "Test feature count does not match "
            "test split length: "
            f"{len(test_x)} vs {len(test)}"
        )

    test_feature_df = test[
        [
            "image_id",
            "patient_id",
            "lesion_id",
            "lesion_uid",
        ]
    ].copy()

    test_feature_df = test_feature_df.merge(
        manifest[
            [
                "image_id",
                "native_diagnosis",
            ]
        ],
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if "feature_index" not in test_feature_df.columns:
        test_feature_df.insert(
            1,
            "feature_index",
            np.arange(len(test_feature_df)),
        )

    if test_feature_df["native_diagnosis"].isna().any():
        raise RuntimeError(
            "Feature/manifest alignment failed "
            "for test features."
        )

    # ------------------------------------------------------------------
    # Training features
    # ------------------------------------------------------------------

    if not TRAIN_FEATURES.exists():
        raise FileNotFoundError(
            "\nTraining feature file does not exist:\n"
            f"  {TRAIN_FEATURES}\n\n"
            "We need to extract C1 training features first.\n"
        )

    train_features = load_feature_file(
    TRAIN_FEATURES
    )

    train_x = train_features["features"]

    if len(train_x) != len(train):
        raise RuntimeError(
            "Training feature count does not match "
            "training split length: "
            f"{len(train_x)} vs {len(train)}"
        )

    train_feature_df = train[
        [
            "image_id",
            "patient_id",
            "lesion_id",
            "lesion_uid",
        ]
    ].copy()

    train_feature_df = train_feature_df.merge(
        manifest[
            [
                "image_id",
                "native_diagnosis",
            ]
        ],
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if "feature_index" not in train_feature_df.columns:
        train_feature_df.insert(
            1,
            "feature_index",
            np.arange(len(train_feature_df)),
        )

    if train_feature_df["native_diagnosis"].isna().any():
        raise RuntimeError(
            "Feature/manifest alignment failed "
            "for training features."
        )

    # ------------------------------------------------------------------
    # Restrict to SCC / BCC
    # ------------------------------------------------------------------

    train_scc = train_feature_df[
        train_feature_df["native_diagnosis"] == "SCC"
    ].copy()

    train_bcc = train_feature_df[
        train_feature_df["native_diagnosis"] == "BCC"
    ].copy()

    test_scc = test_feature_df[
        test_feature_df["native_diagnosis"] == "SCC"
    ].copy()

    print()
    print("FEATURE COVERAGE")
    print(f"Training SCC images: {len(train_scc)}")
    print(f"Training BCC images: {len(train_bcc)}")
    print(f"Test SCC images:     {len(test_scc)}")

    # ------------------------------------------------------------------
    # Mark problematic / clean SCC images
    # ------------------------------------------------------------------

    test_scc["problematic"] = (
        test_scc["image_id"].isin(error_ids)
    )

    test_scc["group"] = np.where(
        test_scc["problematic"],
        "problematic",
        "clean",
    )

    # ------------------------------------------------------------------
    # Feature matrices
    # ------------------------------------------------------------------

    X_train_scc = train_x[
        train_scc["feature_index"].to_numpy()
    ]

    X_train_bcc = train_x[
        train_bcc["feature_index"].to_numpy()
    ]

    X_test_scc = test_x[
        test_scc["feature_index"].to_numpy()
    ]

    # ------------------------------------------------------------------
    # Nearest-neighbor distances
    # ------------------------------------------------------------------

    print()
    print("COMPUTING NEAREST-NEIGHBOR DISTANCES")

    dist_scc = cdist(
        X_test_scc,
        X_train_scc,
        metric="euclidean",
    )

    dist_bcc = cdist(
        X_test_scc,
        X_train_bcc,
        metric="euclidean",
    )

    test_scc["nearest_train_SCC"] = (
        dist_scc.min(axis=1)
    )

    test_scc["nearest_train_BCC"] = (
        dist_bcc.min(axis=1)
    )

    test_scc["nearest_margin"] = (
        test_scc["nearest_train_BCC"]
        - test_scc["nearest_train_SCC"]
    )

    # Positive:
    # closer to training SCC than training BCC.
    #
    # Negative:
    # closer to training BCC than training SCC.

    # ------------------------------------------------------------------
    # Training SCC centroid
    # ------------------------------------------------------------------

    train_scc_centroid = X_train_scc.mean(
        axis=0
    )

    train_bcc_centroid = X_train_bcc.mean(
        axis=0
    )

    test_scc["distance_to_train_SCC_centroid"] = (
        np.linalg.norm(
            X_test_scc
            - train_scc_centroid,
            axis=1,
        )
    )

    test_scc["distance_to_train_BCC_centroid"] = (
        np.linalg.norm(
            X_test_scc
            - train_bcc_centroid,
            axis=1,
        )
    )

    test_scc["centroid_margin"] = (
        test_scc[
            "distance_to_train_BCC_centroid"
        ]
        -
        test_scc[
            "distance_to_train_SCC_centroid"
        ]
    )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TEST SCC COVERAGE SUMMARY")
    print("=" * 80)

    for group in ("problematic", "clean"):
        subset = test_scc[
            test_scc["group"] == group
        ]

        print()
        print(
            f"{group.upper()} SCC:"
        )
        print(
            f"  Images: {len(subset)}"
        )
        print(
            f"  Unique lesions: "
            f"{subset['lesion_uid'].nunique()}"
        )
        print(
            f"  Nearest train SCC: "
            f"{subset['nearest_train_SCC'].mean():.4f}"
        )
        print(
            f"  Nearest train BCC: "
            f"{subset['nearest_train_BCC'].mean():.4f}"
        )
        print(
            f"  Nearest-neighbor margin: "
            f"{subset['nearest_margin'].mean():.4f}"
        )
        print(
            f"  SCC centroid distance: "
            f"{subset['distance_to_train_SCC_centroid'].mean():.4f}"
        )
        print(
            f"  BCC centroid distance: "
            f"{subset['distance_to_train_BCC_centroid'].mean():.4f}"
        )
        print(
            f"  Centroid margin: "
            f"{subset['centroid_margin'].mean():.4f}"
        )

    # ------------------------------------------------------------------
    # Statistical comparison
    # ------------------------------------------------------------------

    problematic = test_scc[
        test_scc["group"] == "problematic"
    ]

    clean = test_scc[
        test_scc["group"] == "clean"
    ]

    print()
    print("=" * 80)
    print("MANN-WHITNEY COMPARISONS")
    print("=" * 80)

    metrics = [
        "nearest_train_SCC",
        "nearest_margin",
        "distance_to_train_SCC_centroid",
        "centroid_margin",
    ]

    results = []

    for metric in metrics:
        u, p = mannwhitneyu(
            problematic[metric],
            clean[metric],
            alternative="two-sided",
        )

        print()
        print(metric)
        print(
            f"  problematic mean: "
            f"{problematic[metric].mean():.6f}"
        )
        print(
            f"  clean mean:       "
            f"{clean[metric].mean():.6f}"
        )
        print(
            f"  U: {u:.6f}"
        )
        print(
            f"  p: {p:.6f}"
        )

        results.append(
            {
                "metric": metric,
                "problematic_mean": problematic[
                    metric
                ].mean(),
                "clean_mean": clean[
                    metric
                ].mean(),
                "U": u,
                "p": p,
            }
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    table_path = (
        OUTPUT_DIR
        / "test_scc_training_coverage.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "summary.txt"
    )

    stats_path = (
        OUTPUT_DIR
        / "statistical_comparison.csv"
    )

    test_scc.to_csv(
        table_path,
        index=False,
    )

    pd.DataFrame(results).to_csv(
        stats_path,
        index=False,
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "DERMASENSE SCC TRAINING-COVERAGE ANALYSIS\n"
        )
        f.write("=" * 80 + "\n")
        f.write(
            f"Training SCC images: {len(train_scc)}\n"
        )
        f.write(
            f"Training SCC lesions: "
            f"{train_scc['lesion_uid'].nunique()}\n"
        )
        f.write(
            f"Training SCC patients: "
            f"{train_scc['patient_id'].nunique()}\n"
        )
        f.write(
            f"Training BCC images: {len(train_bcc)}\n"
        )
        f.write(
            f"Test SCC images: {len(test_scc)}\n"
        )
        f.write(
            f"Problematic SCC images: "
            f"{len(problematic)}\n"
        )
        f.write(
            f"Clean SCC images: "
            f"{len(clean)}\n"
        )

        f.write("\n")
        f.write(
            "STATISTICAL COMPARISON\n"
        )
        f.write(
            pd.DataFrame(results).to_string(
                index=False
            )
        )
        f.write("\n")

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)
    print(f"Table:   {table_path}")
    print(f"Stats:   {stats_path}")
    print(f"Summary: {summary_path}")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
