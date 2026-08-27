from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

from src.data.torch_dataset import CVDatasetTorch
from src.models.native_classifier import (
    DermaSenseNativeClassifier,
    NativeClassifierConfig,
)


PAD_CLASSES = (
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
)

DATASET_CLASSES = {
    "pad_ufes": PAD_CLASSES,
    "isic2019": (
        "AK",
        "BCC",
        "BKL",
        "DF",
        "MEL",
        "NV",
        "SCC",
        "VASC",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SCC/BCC feature geometry using "
            "a trained DermaSense ResNet-50 checkpoint."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="C1 checkpoint to analyze.",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="PAD-UFES split to analyze.",
    )

    parser.add_argument(
        "--output-dir",
        default="analysis/scc_bcc",
    )

    return parser.parse_args()


def resolve_device(requested: str):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def build_dataset(dataset_id: str, split: str):
    return CVDatasetTorch(
        dataset_id=dataset_id,
        split=split,
        verify_images=True,
    )


def collate_images_and_targets(batch):
    return {
        "image": torch.stack(
            [item["image"] for item in batch]
        ),
        "target": torch.tensor(
            [item["target"] for item in batch],
            dtype=torch.long,
        ),
    }


def build_loader(
    dataset,
    batch_size: int,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_images_and_targets,
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
):
    model_config = NativeClassifierConfig(
        backbone="resnet50",
        pretrained=False,
        dropout=0.0,
    )

    model = DermaSenseNativeClassifier(
        model_config
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict"),
    )

    if state_dict is None:
        raise RuntimeError(
            "Could not find model state dict "
            "in checkpoint."
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


def extract_backbone_features(
    model,
    loader,
    device,
):
    """
    Extract the 2048-dimensional ResNet-50 feature
    immediately before the PAD-UFES classification head.

    The classifier itself is not used for the feature
    geometry analysis.
    """

    all_features = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            targets = batch["target"].cpu()

            # DermaSense's backbone produces the feature
            # representation used by the dataset-specific head.
            features = model.forward_features(
                images
            )

            if features.ndim > 2:
                features = torch.flatten(
                    features,
                    start_dim=1,
                )

            all_features.append(
                features.cpu()
            )

            all_targets.append(targets)

    features = torch.cat(
        all_features,
        dim=0,
    ).numpy()

    targets = torch.cat(
        all_targets,
        dim=0,
    ).numpy()

    return features, targets


def summarize_class_features(
    features,
    targets,
    class_names,
):
    if "BCC" not in class_names:
        raise RuntimeError(
            f"BCC is not present in class space: "
            f"{class_names}"
        )

    if "SCC" not in class_names:
        raise RuntimeError(
            f"SCC is not present in class space: "
            f"{class_names}"
        )

    bcc_index = class_names.index("BCC")
    scc_index = class_names.index("SCC")

    results = {}

    for class_name, class_index in (
        ("BCC", bcc_index),
        ("SCC", scc_index),
    ):
        mask = targets == class_index

        class_features = features[mask]

        if len(class_features) == 0:
            raise RuntimeError(
                f"No {class_name} samples found."
            )

        results[class_name] = {
            "count": int(len(class_features)),
            "features": class_features,
            "centroid": class_features.mean(
                axis=0
            ),
        }

    return results


def mean_pairwise_distance(
    features_a,
    features_b,
):
    if len(features_a) == 0 or len(features_b) == 0:
        return float("nan")

    distances = pairwise_distances(
        features_a,
        features_b,
        metric="euclidean",
    )

    return float(distances.mean())


def mean_within_class_distance(
    features,
):
    if len(features) < 2:
        return float("nan")

    distances = pairwise_distances(
        features,
        metric="euclidean",
    )

    upper_triangle = distances[
        np.triu_indices(
            len(features),
            k=1,
        )
    ]

    return float(upper_triangle.mean())


def centroid_distance(
    centroid_a,
    centroid_b,
):
    return float(
        np.linalg.norm(
            centroid_a - centroid_b
        )
    )


def analyze_domain(
    dataset_id: str,
    split: str,
    model,
    device,
    batch_size: int,
):
    print()
    print("=" * 70)
    print(
        f"FEATURE EXTRACTION: "
        f"{dataset_id} / {split}"
    )
    print("=" * 70)

    dataset = build_dataset(
        dataset_id,
        split,
    )

    loader = build_loader(
        dataset,
        batch_size,
    )

    print(
        f"Samples: {len(dataset)}"
    )

    features, targets = (
        extract_backbone_features(
            model,
            loader,
            device,
        )
    )

    print(
        f"Feature matrix: {features.shape}"
    )

    class_names = DATASET_CLASSES[
        dataset_id
    ]

    class_data = summarize_class_features(
        features,
        targets,
        class_names,
    )

    bcc = class_data["BCC"]["features"]
    scc = class_data["SCC"]["features"]

    bcc_within = mean_within_class_distance(
        bcc
    )

    scc_within = mean_within_class_distance(
        scc
    )

    bcc_scc_distance = mean_pairwise_distance(
        bcc,
        scc,
    )

    centroid_distance_value = centroid_distance(
        class_data["BCC"]["centroid"],
        class_data["SCC"]["centroid"],
    )

    mean_within = (
        bcc_within + scc_within
    ) / 2.0

    separation_ratio = (
        bcc_scc_distance / mean_within
        if mean_within > 0
        else float("nan")
    )

    centroid_ratio = (
        centroid_distance_value / mean_within
        if mean_within > 0
        else float("nan")
    )

    result = {
        "dataset": dataset_id,
        "split": split,
        "feature_dimension": int(
            features.shape[1]
        ),
        "bcc_count": int(len(bcc)),
        "scc_count": int(len(scc)),
        "bcc_within_class_distance": bcc_within,
        "scc_within_class_distance": scc_within,
        "bcc_scc_mean_pairwise_distance": (
            bcc_scc_distance
        ),
        "bcc_scc_centroid_distance": (
            centroid_distance_value
        ),
        "mean_within_class_distance": (
            mean_within
        ),
        "pairwise_separation_ratio": (
            separation_ratio
        ),
        "centroid_separation_ratio": (
            centroid_ratio
        ),
        "features": features,
        "targets": targets,
    }

    print()
    print("SCC / BCC FEATURE GEOMETRY")

    print(
        f"BCC samples:                  "
        f"{len(bcc)}"
    )

    print(
        f"SCC samples:                  "
        f"{len(scc)}"
    )

    print(
        f"BCC within-class distance:    "
        f"{bcc_within:.4f}"
    )

    print(
        f"SCC within-class distance:    "
        f"{scc_within:.4f}"
    )

    print(
        f"SCC-BCC pairwise distance:    "
        f"{bcc_scc_distance:.4f}"
    )

    print(
        f"SCC-BCC centroid distance:    "
        f"{centroid_distance_value:.4f}"
    )

    print(
        f"Mean within-class distance:    "
        f"{mean_within:.4f}"
    )

    print(
        f"Pairwise separation ratio:     "
        f"{separation_ratio:.4f}"
    )

    print(
        f"Centroid separation ratio:     "
        f"{centroid_ratio:.4f}"
    )

    return result


def save_domain_features(
    result,
    output_dir: Path,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / f"{result['dataset']}_"
        f"{result['split']}_scc_bcc_features.npz"
    )

    np.savez_compressed(
        path,
        features=result["features"],
        targets=result["targets"],
    )

    return path


def save_summary(
    results,
    checkpoint,
    output_dir: Path,
):
    path = (
        output_dir
        / "scc_bcc_feature_geometry.txt"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "DERMASENSE SCC/BCC FEATURE "
            "GEOMETRY ANALYSIS\n"
        )
        handle.write(
            "=" * 70 + "\n\n"
        )

        handle.write(
            f"Checkpoint experiment: "
            f"{checkpoint.get('experiment')}\n"
        )

        handle.write(
            f"Architecture: "
            f"{checkpoint.get('architecture')}\n"
        )

        handle.write(
            f"Seed: "
            f"{checkpoint.get('seed')}\n"
        )

        handle.write("\n")

        for result in results:
            handle.write(
                f"{result['dataset']} / "
                f"{result['split']}\n"
            )
            handle.write(
                "-" * 50 + "\n"
            )

            fields = (
                (
                    "feature_dimension",
                    "Feature dimension",
                ),
                (
                    "bcc_count",
                    "BCC count",
                ),
                (
                    "scc_count",
                    "SCC count",
                ),
                (
                    "bcc_within_class_distance",
                    "BCC within-class distance",
                ),
                (
                    "scc_within_class_distance",
                    "SCC within-class distance",
                ),
                (
                    "bcc_scc_mean_pairwise_distance",
                    "SCC-BCC mean pairwise distance",
                ),
                (
                    "bcc_scc_centroid_distance",
                    "SCC-BCC centroid distance",
                ),
                (
                    "mean_within_class_distance",
                    "Mean within-class distance",
                ),
                (
                    "pairwise_separation_ratio",
                    "Pairwise separation ratio",
                ),
                (
                    "centroid_separation_ratio",
                    "Centroid separation ratio",
                ),
            )

            for key, label in fields:
                handle.write(
                    f"{label}: "
                    f"{result[key]}\n"
                )

            handle.write("\n")

    return path


def main():
    args = parse_args()

    device = resolve_device(
        args.device
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    output_dir = Path(
        args.output_dir
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    print("=" * 70)
    print("DERMASENSE SCC/BCC FEATURE ANALYSIS")
    print("=" * 70)

    print(
        f"Device:     {device}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    model, checkpoint = load_model(
        checkpoint_path,
        device,
    )

    print(
        f"Experiment: "
        f"{checkpoint.get('experiment')}"
    )

    print(
        f"Architecture: "
        f"{checkpoint.get('architecture')}"
    )

    print(
        f"Seed: "
        f"{checkpoint.get('seed')}"
    )

    results = []

    # PAD-UFES is the primary domain.
    results.append(
        analyze_domain(
            "pad_ufes",
            args.split,
            model,
            device,
            args.batch_size,
        )
    )

    # ISIC comparison is useful for domain-shift
    # analysis. The dataset split must exist in the
    # current repository configuration.
    try:
        results.append(
            analyze_domain(
                "isic2019",
                "test",
                model,
                device,
                args.batch_size,
            )
        )
    except Exception as exc:
        print()
        print(
            "WARNING: Could not analyze "
            "ISIC2019 test split."
        )
        print(
            f"Reason: {exc}"
        )

    print()
    print("=" * 70)
    print("DOMAIN COMPARISON")
    print("=" * 70)

    for result in results:
        print(
            f"{result['dataset']:>12} | "
            f"pairwise ratio="
            f"{result['pairwise_separation_ratio']:.4f} | "
            f"centroid ratio="
            f"{result['centroid_separation_ratio']:.4f}"
        )

    for result in results:
        feature_path = save_domain_features(
            result,
            output_dir,
        )

        print(
            f"Saved features: "
            f"{feature_path}"
        )

    summary_path = save_summary(
        results,
        checkpoint,
        output_dir,
    )

    print(
        f"Saved summary: "
        f"{summary_path}"
    )

    print()
    print("=" * 70)
    print("SCC/BCC FEATURE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
