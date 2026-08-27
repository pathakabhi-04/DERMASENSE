from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


PAD_CLASSES = (
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
)

BCC_INDEX = PAD_CLASSES.index("BCC")
SCC_INDEX = PAD_CLASSES.index("SCC")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SCC lesion-level feature geometry."
        )
    )

    parser.add_argument(
        "--features",
        default=(
            "analysis/scc_bcc/"
            "pad_ufes_test_scc_bcc_features.npz"
        ),
        help=(
            "NPZ file containing feature vectors and targets."
        ),
    )

    parser.add_argument(
        "--errors",
        default=(
            "analysis/scc_bcc/c1_seed42_errors/"
            "scc_to_bcc_errors.csv"
        ),
        help=(
            "CSV containing SCC-to-BCC error image IDs."
        ),
    )

    parser.add_argument(
        "--manifest",
        default=(
            "data/splits/pad_ufes/test.csv"
        ),
        help=(
            "PAD-UFES split manifest used to align "
            "features with image metadata."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "analysis/scc_bcc/"
            "c1_lesion_geometry"
        ),
        help=(
            "Directory for the lesion-level outputs."
        ),
    )

    return parser.parse_args()


def load_data(args):
    feature_data = np.load(
        args.features,
        allow_pickle=True,
    )

    features = feature_data["features"]
    targets = feature_data["targets"]

    if features.ndim != 2:
        raise RuntimeError(
            f"Expected 2-D features, got {features.shape}."
        )

    if targets.ndim != 1:
        raise RuntimeError(
            f"Expected 1-D targets, got {targets.shape}."
        )

    if len(features) != len(targets):
        raise RuntimeError(
            "Feature and target lengths differ: "
            f"{len(features)} vs {len(targets)}."
        )

    errors = {}

    with open(
        args.errors,
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(
                "Error CSV has no header."
            )

        if "image_id" not in reader.fieldnames:
            raise RuntimeError(
                "Error CSV does not contain "
                "'image_id'."
            )

        for row in reader:
            errors[str(row["image_id"])] = row

    import pandas as pd

    manifest = pd.read_csv(
        args.manifest
    )

    required_columns = (
        "image_id",
        "patient_id",
        "lesion_uid",
        "native_diagnosis",
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Manifest is missing required columns: "
            f"{missing_columns}"
        )

    return (
        features,
        targets,
        manifest,
        errors,
    )


def pairwise_mean_distance(
    vectors_a,
    vectors_b,
):
    if len(vectors_a) == 0:
        return float("nan")

    if len(vectors_b) == 0:
        return float("nan")

    differences = (
        vectors_a[:, None, :]
        - vectors_b[None, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=2,
    )

    return float(
        distances.mean()
    )


def main():
    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("DERMASENSE SCC LESION-LEVEL FEATURE GEOMETRY")
    print("=" * 80)

    (
        features,
        targets,
        manifest,
        errors,
    ) = load_data(args)

    print()
    print(
        f"Feature matrix: {features.shape}"
    )

    # ------------------------------------------------------------
    # Restrict to SCC/BCC features.
    # ------------------------------------------------------------

    bcc_mask = (
        targets == BCC_INDEX
    )

    scc_mask = (
        targets == SCC_INDEX
    )

    bcc_features = features[
        bcc_mask
    ]

    scc_features = features[
        scc_mask
    ]

    print(
        f"BCC images: {len(bcc_features)}"
    )

    print(
        f"SCC images: {len(scc_features)}"
    )

    if len(bcc_features) == 0:
        raise RuntimeError(
            "No BCC features found."
        )

    if len(scc_features) == 0:
        raise RuntimeError(
            "No SCC features found."
        )

    # ------------------------------------------------------------
    # Global class centroids.
    # ------------------------------------------------------------

    bcc_centroid = (
        bcc_features.mean(axis=0)
    )

    scc_centroid = (
        scc_features.mean(axis=0)
    )

    print()
    print("GLOBAL CENTROIDS")

    print(
        "BCC centroid computed."
    )

    print(
        "SCC centroid computed."
    )

    # ------------------------------------------------------------
    # Build image -> feature metadata mapping.
    #
    # The feature file was produced from the same dataset
    # ordering as the manifest, so manifest ordering is used
    # to associate each feature vector with its image metadata.
    # ------------------------------------------------------------

    if len(manifest) != len(features):
        raise RuntimeError(
            "Manifest and feature matrix lengths differ: "
            f"{len(manifest)} vs {len(features)}"
        )

    image_ids = (
        manifest["image_id"]
        .astype(str)
        .to_numpy()
    )

    patient_ids = (
        manifest["patient_id"]
        .astype(str)
        .to_numpy()
    )

    lesion_uids = (
        manifest["lesion_uid"]
        .astype(str)
        .to_numpy()
    )

    diagnoses = (
        manifest["native_diagnosis"]
        .astype(str)
        .to_numpy()
    )

    # ------------------------------------------------------------
    # Verify feature/manifest alignment.
    # ------------------------------------------------------------

    for index in range(
        len(image_ids)
    ):
        diagnosis = diagnoses[index]

        if diagnosis not in PAD_CLASSES:
            raise RuntimeError(
                "Unknown diagnosis in manifest "
                f"at index {index}: {diagnosis!r}"
            )

        expected_target = (
            PAD_CLASSES.index(
                diagnosis
            )
        )

        if int(targets[index]) != expected_target:
            raise RuntimeError(
                "Feature/manifest target mismatch "
                f"at index {index}: "
                f"feature target={targets[index]}, "
                f"manifest diagnosis="
                f"{diagnosis!r}"
            )

    print()
    print(
        "Feature/manifest alignment: PASS"
    )

    # ------------------------------------------------------------
    # SCC lesion analysis.
    # ------------------------------------------------------------

    scc_indices = np.where(
        scc_mask
    )[0]

    problematic_image_ids = set(
        errors.keys()
    )

    rows = []

    unique_lesions = sorted(
        set(
            lesion_uids[
                scc_indices
            ]
        )
    )

    for lesion_uid in unique_lesions:
        lesion_indices = scc_indices[
            lesion_uids[
                scc_indices
            ]
            == lesion_uid
        ]

        lesion_features = features[
            lesion_indices
        ]

        lesion_image_ids = image_ids[
            lesion_indices
        ]

        lesion_patient_ids = patient_ids[
            lesion_indices
        ]

        lesion_diagnoses = diagnoses[
            lesion_indices
        ]

        # Every lesion must belong to exactly one patient.
        if len(
            set(lesion_patient_ids)
        ) != 1:
            raise RuntimeError(
                f"Lesion {lesion_uid} "
                "belongs to multiple patients."
            )

        # Every lesion analyzed here must be SCC.
        if set(lesion_diagnoses) != {"SCC"}:
            raise RuntimeError(
                f"Lesion {lesion_uid} contains "
                "non-SCC diagnoses."
            )

        patient_id = (
            lesion_patient_ids[0]
        )

        lesion_centroid = (
            lesion_features.mean(
                axis=0
            )
        )

        distance_to_bcc = float(
            np.linalg.norm(
                lesion_centroid
                - bcc_centroid
            )
        )

        distance_to_scc = float(
            np.linalg.norm(
                lesion_centroid
                - scc_centroid
            )
        )

        # --------------------------------------------------------
        # Signed centroid margin.
        #
        # Definition:
        #
        #     bcc_minus_scc_distance
        #       = distance_to_SCC - distance_to_BCC
        #
        # Therefore:
        #
        #   positive -> BCC centroid is closer
        #   negative -> SCC centroid is closer
        #   zero     -> equally distant
        # --------------------------------------------------------

        bcc_minus_scc_distance = (
            distance_to_scc
            - distance_to_bcc
        )

        error_image_ids = [
            image_id
            for image_id in lesion_image_ids
            if image_id in problematic_image_ids
        ]

        error_count = len(
            error_image_ids
        )

        rows.append(
            {
                "patient_id": patient_id,
                "lesion_uid": lesion_uid,
                "image_count": len(
                    lesion_indices
                ),
                "scc_to_bcc_error_images": (
                    error_count
                ),
                "error_fraction": (
                    error_count
                    / len(lesion_indices)
                ),
                "distance_to_bcc_centroid": (
                    distance_to_bcc
                ),
                "distance_to_scc_centroid": (
                    distance_to_scc
                ),
                "bcc_minus_scc_distance": (
                    bcc_minus_scc_distance
                ),
                "image_ids": ";".join(
                    lesion_image_ids
                ),
                "error_image_ids": ";".join(
                    error_image_ids
                ),
            }
        )

    import pandas as pd

    lesion_df = pd.DataFrame(
        rows
    )

    lesion_df = lesion_df.sort_values(
        [
            "scc_to_bcc_error_images",
            "bcc_minus_scc_distance",
        ],
        ascending=[
            False,
            False,
        ],
    )

    # ------------------------------------------------------------
    # Classify lesions by whether at least one image was
    # misclassified from SCC to BCC.
    # ------------------------------------------------------------

    lesion_df["error_status"] = np.where(
        lesion_df[
            "scc_to_bcc_error_images"
        ]
        > 0,
        "SCC_to_BCC_error",
        "clean_SCC",
    )

    problematic = lesion_df[
        lesion_df["error_status"]
        == "SCC_to_BCC_error"
    ]

    clean = lesion_df[
        lesion_df["error_status"]
        == "clean_SCC"
    ]

    # ------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("LESION SUMMARY")
    print("=" * 80)

    print(
        f"Total SCC lesions: "
        f"{len(lesion_df)}"
    )

    print(
        f"Problematic SCC lesions: "
        f"{len(problematic)}"
    )

    print(
        f"Clean SCC lesions: "
        f"{len(clean)}"
    )

    print()
    print("=" * 80)
    print("DISTANCE TO CLASS CENTROIDS")
    print("=" * 80)

    for name, group in (
        ("Problematic", problematic),
        ("Clean", clean),
    ):
        print()
        print(name)

        print(
            f"  N lesions: "
            f"{len(group)}"
        )

        print(
            f"  Mean distance → BCC: "
            f"{group['distance_to_bcc_centroid'].mean():.4f}"
        )

        print(
            f"  Mean distance → SCC: "
            f"{group['distance_to_scc_centroid'].mean():.4f}"
        )

        print(
            f"  Mean BCC-minus-SCC distance: "
            f"{group['bcc_minus_scc_distance'].mean():.4f}"
        )

        print(
            f"  Median BCC-minus-SCC distance: "
            f"{group['bcc_minus_scc_distance'].median():.4f}"
        )

    # ------------------------------------------------------------
    # Centroid-side analysis.
    #
    # Positive BCC-minus-SCC distance means the lesion is
    # closer to the BCC centroid.
    # ------------------------------------------------------------

    problematic_bcc_closer = problematic[
        problematic[
            "bcc_minus_scc_distance"
        ]
        > 0
    ]

    clean_bcc_closer = clean[
        clean[
            "bcc_minus_scc_distance"
        ]
        > 0
    ]

    problematic_scc_closer = problematic[
        problematic[
            "bcc_minus_scc_distance"
        ]
        < 0
    ]

    clean_scc_closer = clean[
        clean[
            "bcc_minus_scc_distance"
        ]
        < 0
    ]

    print()
    print("=" * 80)
    print("CENTROID-SIDE ANALYSIS")
    print("=" * 80)

    print(
        "Positive margin = closer to BCC."
    )

    print(
        "Negative margin = closer to SCC."
    )

    print(
        "Problematic SCC lesions closer "
        "to BCC centroid: "
        f"{len(problematic_bcc_closer)}/"
        f"{len(problematic)}"
    )

    print(
        "Problematic SCC lesions closer "
        "to SCC centroid: "
        f"{len(problematic_scc_closer)}/"
        f"{len(problematic)}"
    )

    print(
        "Clean SCC lesions closer "
        "to BCC centroid: "
        f"{len(clean_bcc_closer)}/"
        f"{len(clean)}"
    )

    print(
        "Clean SCC lesions closer "
        "to SCC centroid: "
        f"{len(clean_scc_closer)}/"
        f"{len(clean)}"
    )

    # ------------------------------------------------------------
    # Save CSV.
    # ------------------------------------------------------------

    csv_path = (
        output_dir
        / "scc_lesion_geometry.csv"
    )

    lesion_df.to_csv(
        csv_path,
        index=False,
    )

    print()
    print(
        f"Saved lesion table: "
        f"{csv_path}"
    )

    # ------------------------------------------------------------
    # Save concise summary.
    # ------------------------------------------------------------

    summary_path = (
        output_dir
        / "summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "DERMASENSE SCC LESION GEOMETRY\n"
        )

        handle.write(
            "=" * 60 + "\n\n"
        )

        handle.write(
            "Centroid margin definition:\n"
        )

        handle.write(
            "  BCC-minus-SCC distance = "
            "distance_to_SCC - distance_to_BCC\n"
        )

        handle.write(
            "  Positive = closer to BCC\n"
        )

        handle.write(
            "  Negative = closer to SCC\n\n"
        )

        handle.write(
            f"Total SCC lesions: "
            f"{len(lesion_df)}\n"
        )

        handle.write(
            f"Problematic lesions: "
            f"{len(problematic)}\n"
        )

        handle.write(
            f"Clean lesions: "
            f"{len(clean)}\n\n"
        )

        for name, group in (
            ("Problematic", problematic),
            ("Clean", clean),
        ):
            handle.write(
                f"{name} SCC lesions\n"
            )

            handle.write(
                f"  N: {len(group)}\n"
            )

            handle.write(
                "  Mean distance to BCC: "
                f"{group['distance_to_bcc_centroid'].mean():.6f}\n"
            )

            handle.write(
                "  Mean distance to SCC: "
                f"{group['distance_to_scc_centroid'].mean():.6f}\n"
            )

            handle.write(
                "  Mean BCC-minus-SCC distance: "
                f"{group['bcc_minus_scc_distance'].mean():.6f}\n"
            )

            handle.write(
                "  Median BCC-minus-SCC distance: "
                f"{group['bcc_minus_scc_distance'].median():.6f}\n\n"
            )

        handle.write(
            "Problematic lesions closer to BCC: "
            f"{len(problematic_bcc_closer)}/"
            f"{len(problematic)}\n"
        )

        handle.write(
            "Problematic lesions closer to SCC: "
            f"{len(problematic_scc_closer)}/"
            f"{len(problematic)}\n"
        )

        handle.write(
            "Clean lesions closer to BCC: "
            f"{len(clean_bcc_closer)}/"
            f"{len(clean)}\n"
        )

        handle.write(
            "Clean lesions closer to SCC: "
            f"{len(clean_scc_closer)}/"
            f"{len(clean)}\n"
        )

    print(
        f"Saved summary: {summary_path}"
    )

    print()
    print("=" * 80)
    print("LESION-LEVEL ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()