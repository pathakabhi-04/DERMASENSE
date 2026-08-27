from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from torch.utils.data import DataLoader

from src.data.loader import (
    DataLoaderConfig,
    build_dataloader,
)
from src.data.torch_dataset import CVDatasetTorch
from src.data.transforms import (
    ImageTransformConfig,
    build_eval_transform,
)
from src.models.native_classifier import (
    DermaSenseNativeClassifier,
    NativeClassifierConfig,
)
from src.training.checkpoint import load_checkpoint


TARGET_CLASSES = ("BCC", "MEL", "SCC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment A: frozen ISIC feature transfer "
            "to PAD-UFES-20."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="ISIC experiment configuration.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Frozen ISIC checkpoint.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration file must contain a YAML mapping."
        )

    return config


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    return requested


def build_model(config: dict) -> DermaSenseNativeClassifier:
    model_config = NativeClassifierConfig(
        backbone=config["model"]["backbone"],
        pretrained=False,
        dropout=config["model"].get(
            "dropout",
            0.0,
        ),
    )

    return DermaSenseNativeClassifier(model_config)


def build_dataset(
    dataset_id: str,
    split: str,
) -> CVDatasetTorch:
    transform_config = ImageTransformConfig()

    return CVDatasetTorch(
        dataset_id=dataset_id,
        split=split,
        transform=build_eval_transform(
            transform_config
        ),
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


def build_loader(dataset, batch_size: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_images_and_targets,
    )

@torch.no_grad()
def extract_features(
    model: DermaSenseNativeClassifier,
    loader,
    device: str,
):
    features = []
    labels = []
    diagnoses = []

    model.eval()

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        batch_features = model.forward_features(
            images
        )

        features.append(
            batch_features.cpu()
        )

        labels.append(
            batch["target"].cpu()
        )

        diagnoses.extend(
            sample.native_diagnosis
            for sample in batch["sample"]
        )

    return (
        torch.cat(features, dim=0),
        torch.cat(labels, dim=0),
        diagnoses,
    )


def normalize_features(
    features: torch.Tensor,
) -> torch.Tensor:
    return torch.nn.functional.normalize(
        features,
        p=2,
        dim=1,
    )


def cosine_distances_to_centroid(
    features: torch.Tensor,
    centroid: torch.Tensor,
) -> torch.Tensor:
    features = normalize_features(features)

    centroid = torch.nn.functional.normalize(
        centroid.unsqueeze(0),
        p=2,
        dim=1,
    ).squeeze(0)

    return 1.0 - torch.matmul(
        features,
        centroid,
    )


def make_centroids(
    features: torch.Tensor,
    diagnoses: list[str],
) -> dict[str, torch.Tensor]:
    normalized = normalize_features(features)

    centroids = {}

    for class_name in TARGET_CLASSES:
        indices = [
            i
            for i, diagnosis in enumerate(diagnoses)
            if diagnosis == class_name
        ]

        if not indices:
            raise RuntimeError(
                f"No ISIC training samples found for "
                f"class {class_name}."
            )

        centroid = normalized[indices].mean(dim=0)

        centroids[class_name] = (
            torch.nn.functional.normalize(
                centroid,
                p=2,
                dim=0,
            )
        )

        print(
            f"ISIC centroid {class_name}: "
            f"{len(indices)} samples"
        )

    return centroids


def summarize_distances(
    distances: torch.Tensor,
) -> dict[str, float]:
    return {
        "mean": distances.mean().item(),
        "median": distances.median().item(),
        "std": distances.std(
            unbiased=False
        ).item(),
        "min": distances.min().item(),
        "max": distances.max().item(),
    }


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)

    print("=" * 70)
    print("DERMASENSE EXPERIMENT A")
    print("FROZEN FEATURE TRANSFER: ISIC → PAD-UFES")
    print("=" * 70)

    print(f"Device:      {device}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(
        f"Backbone:    "
        f"{config['model']['backbone']}"
    )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{args.checkpoint}"
        )

    # --------------------------------------------------------------
    # Build frozen ISIC model
    # --------------------------------------------------------------

    model = build_model(config)

    metadata = load_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=None,
        expected_dataset_id="isic2019",
        expected_num_classes=8,
        expected_architecture=config[
            "experiment"
        ]["architecture"],
        map_location=device,
    )

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    print()
    print("CHECKPOINT")
    print(
        f"Epoch:       {metadata.epoch}"
    )
    print(
        f"Best val F1: {metadata.val_macro_f1:.4f}"
    )

    # --------------------------------------------------------------
    # ISIC TRAIN — reference representation
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("ISIC TRAIN FEATURE EXTRACTION")
    print("=" * 70)

    isic_train = build_dataset(
        "isic2019",
        "train",
    )

    isic_loader = build_loader(
        isic_train,
        args.batch_size,
    )

    isic_features, isic_labels, isic_diagnoses = (
        extract_features(
            model,
            isic_loader,
            device,
        )
    )

    print(
        f"ISIC samples:       {len(isic_features)}"
    )
    print(
        f"Feature dimension:  {isic_features.shape[1]}"
    )

    centroids = make_centroids(
        isic_features,
        isic_diagnoses,
    )

    # --------------------------------------------------------------
    # ISIC within-class reference distances
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("ISIC REFERENCE DISTANCES")
    print("=" * 70)

    for class_name in TARGET_CLASSES:
        mask = torch.tensor(
            [
                diagnosis == class_name
                for diagnosis in isic_diagnoses
            ],
            dtype=torch.bool,
        )

        class_features = isic_features[mask]

        distances = cosine_distances_to_centroid(
            class_features,
            centroids[class_name],
        )

        stats = summarize_distances(
            distances
        )

        print(
            f"{class_name:>3} | "
            f"n={mask.sum().item():4d} | "
            f"mean={stats['mean']:.4f} | "
            f"median={stats['median']:.4f} | "
            f"std={stats['std']:.4f}"
        )

    # --------------------------------------------------------------
    # PAD-UFES TEST — transfer analysis
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("PAD-UFES TEST FEATURE EXTRACTION")
    print("=" * 70)

    pad_test = build_dataset(
        "pad_ufes",
        "test",
    )

    pad_loader = build_loader(
        pad_test,
        args.batch_size,
    )

    pad_features, pad_labels, pad_diagnoses = (
        extract_features(
            model,
            pad_loader,
            device,
        )
    )

    print(
        f"PAD-UFES samples:   {len(pad_features)}"
    )
    print(
        f"Feature dimension:  {pad_features.shape[1]}"
    )

    # --------------------------------------------------------------
    # Nearest centroid
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("PAD-UFES → ISIC CENTROID TRANSFER")
    print("=" * 70)

    centroid_matrix = torch.stack(
        [
            centroids[class_name]
            for class_name in TARGET_CLASSES
        ],
        dim=0,
    )

    normalized_pad = normalize_features(
        pad_features
    )

    similarities = torch.matmul(
        normalized_pad,
        centroid_matrix.T,
    )

    nearest_indices = similarities.argmax(
        dim=1
    )

    nearest_predictions = [
        TARGET_CLASSES[index]
        for index in nearest_indices.tolist()
    ]

    evaluated_indices = [
        i
        for i, diagnosis in enumerate(pad_diagnoses)
        if diagnosis in TARGET_CLASSES
    ]

    correct = 0

    for i in evaluated_indices:
        if (
            nearest_predictions[i]
            == pad_diagnoses[i]
        ):
            correct += 1

    print(
        f"Evaluated classes:  {TARGET_CLASSES}"
    )
    print(
        f"Evaluated samples:  {len(evaluated_indices)}"
    )

    if evaluated_indices:
        print(
            f"Nearest-centroid accuracy: "
            f"{correct / len(evaluated_indices):.4f}"
        )

    # --------------------------------------------------------------
    # Per-class PAD distances
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("PAD-UFES PER-CLASS DISTANCES")
    print("=" * 70)

    for class_name in TARGET_CLASSES:
        indices = [
            i
            for i, diagnosis in enumerate(
                pad_diagnoses
            )
            if diagnosis == class_name
        ]

        class_features = pad_features[
            indices
        ]

        distances = (
            cosine_distances_to_centroid(
                class_features,
                centroids[class_name],
            )
        )

        stats = summarize_distances(
            distances
        )

        print(
            f"{class_name:>3} | "
            f"n={len(indices):3d} | "
            f"mean={stats['mean']:.4f} | "
            f"median={stats['median']:.4f} | "
            f"std={stats['std']:.4f} | "
            f"min={stats['min']:.4f} | "
            f"max={stats['max']:.4f}"
        )

    # --------------------------------------------------------------
    # Interpretation aid: PAD vs ISIC reference ratio
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("PAD / ISIC DISTANCE COMPARISON")
    print("=" * 70)

    for class_name in TARGET_CLASSES:
        isic_mask = torch.tensor(
            [
                diagnosis == class_name
                for diagnosis in isic_diagnoses
            ],
            dtype=torch.bool,
        )

        pad_mask = torch.tensor(
            [
                diagnosis == class_name
                for diagnosis in pad_diagnoses
            ],
            dtype=torch.bool,
        )

        isic_distances = (
            cosine_distances_to_centroid(
                isic_features[isic_mask],
                centroids[class_name],
            )
        )

        pad_distances = (
            cosine_distances_to_centroid(
                pad_features[pad_mask],
                centroids[class_name],
            )
        )

        isic_mean = (
            isic_distances.mean().item()
        )

        pad_mean = (
            pad_distances.mean().item()
        )

        ratio = (
            pad_mean / isic_mean
            if isic_mean > 0
            else float("inf")
        )

        print(
            f"{class_name:>3} | "
            f"ISIC mean={isic_mean:.4f} | "
            f"PAD mean={pad_mean:.4f} | "
            f"ratio={ratio:.2f}x"
        )

    print()
    print("=" * 70)
    print("EXPERIMENT A COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
