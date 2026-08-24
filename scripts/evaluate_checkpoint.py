from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.data.loader import (
    DataLoaderConfig,
    build_dataloader,
)
from src.data.torch_dataset import CVDatasetTorch
from src.models.native_classifier import (
    DermaSenseNativeClassifier,
    NativeClassifierConfig,
)
from src.training.checkpoint import load_checkpoint
from src.training.engine import _resolve_device
from src.training.metrics import (
    classification_metrics,
    confusion_matrix,
    per_class_metrics,
)


ARCHITECTURE_VERSION = "CV_MODEL_ARCHITECTURE_v1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a DermaSense CV checkpoint."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to CV experiment configuration.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Dataset split to evaluate.",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration file must contain a YAML mapping."
        )

    return config


def build_dataset(
    dataset_id: str,
    split: str,
) -> CVDatasetTorch:
    return CVDatasetTorch(
        dataset_id=dataset_id,
        split=split,
        verify_images=True,
    )


def build_loader(
    dataset: CVDatasetTorch,
    config: dict,
):
    runtime = config["runtime"]
    training = config["training"]

    loader_config = DataLoaderConfig(
        batch_size=training["batch_size"],
        num_workers=runtime.get(
            "num_workers",
            0,
        ),
        pin_memory=runtime.get(
            "pin_memory",
            False,
        ),
        drop_last=False,
    )

    return build_dataloader(
        dataset=dataset,
        config=loader_config,
    )


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    loader,
    dataset_id: str,
    num_classes: int,
    device: torch.device,
) -> tuple[float, object, torch.Tensor, torch.Tensor]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    all_predictions = []
    all_targets = []

    for batch in loader:
        if "image" not in batch:
            raise RuntimeError(
                "Batch is missing required key: 'image'."
            )

        if "target" not in batch:
            raise RuntimeError(
                "Batch is missing required key: 'target'."
            )

        images = batch["image"].to(device)
        targets = batch["target"].long().to(device)

        logits = model(
            images,
            dataset_id=dataset_id,
        )

        if logits.ndim != 2:
            raise RuntimeError(
                "Expected logits with shape [B, C]. "
                f"Got {tuple(logits.shape)}."
            )

        if logits.shape[1] != num_classes:
            raise RuntimeError(
                "Model output class count does not match "
                f"dataset: {logits.shape[1]} vs "
                f"{num_classes}."
            )

        loss = criterion(
            logits,
            targets,
        )

        predictions = logits.argmax(
            dim=1
        )

        batch_size = targets.shape[0]

        total_loss += (
            loss.item() * batch_size
        )
        total_samples += batch_size

        all_predictions.append(
            predictions.cpu()
        )
        all_targets.append(
            targets.cpu()
        )

    if total_samples == 0:
        raise RuntimeError(
            "Evaluation processed zero samples."
        )

    predictions = torch.cat(
        all_predictions,
        dim=0,
    )

    targets = torch.cat(
        all_targets,
        dim=0,
    )

    metrics = classification_metrics(
        predictions,
        targets,
        num_classes,
    )

    mean_loss = (
        total_loss / total_samples
    )

    return (
        mean_loss,
        metrics,
        predictions,
        targets,
    )


def main() -> None:
    args = parse_args()

    config = load_config(
        args.config
    )

    dataset_id = config["dataset"]["id"]

    if args.split == "val":
        split = config["dataset"]["val_split"]
    else:
        split = config["dataset"]["test_split"]

    device = _resolve_device(
        config["runtime"].get(
            "device",
            "auto",
        )
    )

    print("=" * 70)
    print("DERMASENSE CHECKPOINT EVALUATION")
    print("=" * 70)

    print(
        f"Experiment:    "
        f"{config['experiment']['name']}"
    )
    print(
        f"Architecture:  "
        f"{config['experiment']['architecture']}"
    )
    print(
        f"Dataset:       {dataset_id}"
    )
    print(
        f"Split:         {split}"
    )
    print(
        f"Device:        {device}"
    )
    print(
        f"Checkpoint:    {args.checkpoint}"
    )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{args.checkpoint}"
        )

    dataset = build_dataset(
        dataset_id,
        split,
    )

    loader = build_loader(
        dataset,
        config,
    )

    print()
    print("DATASET")
    print(
        f"Samples:       {len(dataset)}"
    )
    print(
        f"Classes:       {dataset.class_names}"
    )
    print(
        f"Num classes:   {dataset.num_classes}"
    )

    expected_architecture = config[
        "experiment"
    ].get(
        "architecture",
        ARCHITECTURE_VERSION,
    )

    if expected_architecture != ARCHITECTURE_VERSION:
        raise RuntimeError(
            "Experiment architecture identifier does "
            "not match the locked architecture version: "
            f"{expected_architecture!r}"
        )

    model_config = NativeClassifierConfig(
        backbone=config["model"]["backbone"],
        pretrained=False,
        dropout=config["model"].get(
            "dropout",
            0.0,
        ),
    )

    model = DermaSenseNativeClassifier(
        model_config
    )

    metadata = load_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=None,
        expected_dataset_id=dataset_id,
        expected_num_classes=dataset.num_classes,
        expected_architecture=expected_architecture,
        map_location=device,
    )

    model = model.to(device)

    print()
    print("CHECKPOINT")
    print(
        f"Architecture:  "
        f"{metadata.architecture}"
    )
    print(
        f"Dataset:       "
        f"{metadata.dataset_id}"
    )
    print(
        f"Classes:       "
        f"{metadata.num_classes}"
    )
    print(
        f"Epoch:         "
        f"{metadata.epoch}"
    )
    print(
        f"Best val F1:   "
        f"{metadata.val_macro_f1:.4f}"
    )

    (
        loss,
        metrics,
        predictions,
        targets,
    ) = evaluate(
        model=model,
        loader=loader,
        dataset_id=dataset_id,
        num_classes=dataset.num_classes,
        device=device,
    )

    print()
    print("EVALUATION")
    print(
        f"Loss:          {loss:.6f}"
    )
    print(
        f"Accuracy:      "
        f"{metrics.accuracy:.6f}"
    )
    print(
        f"Macro F1:      "
        f"{metrics.macro_f1:.6f}"
    )
    print(
        f"Weighted F1:   "
        f"{metrics.weighted_f1:.6f}"
    )

    matrix = confusion_matrix(
        predictions,
        targets,
        dataset.num_classes,
    )

    print()
    print("CONFUSION MATRIX")
    print(
        "Rows = true class, "
        "Columns = predicted class"
    )
    print()

    header = "          " + " ".join(
        f"{name:>6s}"
        for name in dataset.class_names
    )

    print(header)

    for index, class_name in enumerate(
        dataset.class_names
    ):
        values = " ".join(
            f"{int(value):6d}"
            for value in matrix[index].tolist()
        )

        print(
            f"{class_name:>8s} {values}"
        )

    print()
    print(
        "Per-class metrics:"
    )

    per_class = per_class_metrics(
        predictions,
        targets,
        dataset.num_classes,
    )

    for detail in per_class:
        class_index = int(
            detail["class_index"]
        )

        class_name = dataset.class_names[
            class_index
        ]

        print(
            f"  {class_index:2d} "
            f"{class_name:6s} | "
            f"precision="
            f"{detail['precision']:.4f} | "
            f"recall="
            f"{detail['recall']:.4f} | "
            f"f1="
            f"{detail['f1']:.4f} | "
            f"support="
            f"{int(detail['support'])}"
        )

    if args.split == "test":
        print()
        print(
            "NOTE: This is the frozen test evaluation. "
            "Do not use these results for model selection."
        )

    print()
    print("=" * 70)
    print("CHECKPOINT EVALUATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()