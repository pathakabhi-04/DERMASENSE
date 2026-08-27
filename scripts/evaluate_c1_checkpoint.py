from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluation-only C1 checkpoint evaluator. "
            "Does not train or modify the checkpoint."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="C1 checkpoint to evaluate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
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
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def build_dataset():
    return CVDatasetTorch(
        dataset_id="pad_ufes",
        split="test",
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


def build_loader(dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_images_and_targets,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    targets = []
    predictions = []

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        batch_targets = batch["target"].to(
            device,
            non_blocking=True,
        )

        logits = model(
            images,
            dataset_id="pad_ufes",
        )

        batch_predictions = (
            torch.argmax(
                logits,
                dim=1,
            )
        )

        targets.extend(
            batch_targets.cpu().numpy().tolist()
        )

        predictions.extend(
            batch_predictions.cpu().numpy().tolist()
        )

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    return targets, predictions


def main():
    args = parse_args()

    device = resolve_device(
        args.device
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    print("=" * 70)
    print("DERMASENSE C1 CHECKPOINT EVALUATION")
    print("=" * 70)

    print(
        f"Device:      {device}"
    )

    print(
        f"Checkpoint:  {checkpoint_path}"
    )

    print(
        f"Batch size:  {args.batch_size}"
    )

    print()
    print("=" * 70)
    print("LOADING CHECKPOINT")
    print("=" * 70)

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

    print(
        "Checkpoint loaded successfully."
    )

    if "experiment" in checkpoint:
        print(
            f"Experiment:  "
            f"{checkpoint['experiment']}"
        )

    if "architecture" in checkpoint:
        print(
            f"Architecture: "
            f"{checkpoint['architecture']}"
        )

    if "seed" in checkpoint:
        print(
            f"Seed:         "
            f"{checkpoint['seed']}"
        )

    if "best_epoch" in checkpoint:
        print(
            f"Best epoch:   "
            f"{checkpoint['best_epoch']}"
        )

    if "best_val_macro_f1" in checkpoint:
        print(
            f"Best val F1:  "
            f"{checkpoint['best_val_macro_f1']:.6f}"
        )

    print()
    print("=" * 70)
    print("TEST DATASET")
    print("=" * 70)

    dataset = build_dataset()

    print(
        f"Test samples: {len(dataset)}"
    )

    print(
        f"Classes:      {dataset.class_names}"
    )

    if tuple(dataset.class_names) != PAD_CLASSES:
        raise RuntimeError(
            "Unexpected PAD-UFES class ordering: "
            f"{dataset.class_names}"
        )

    loader = build_loader(
        dataset,
        args.batch_size,
    )

    print()
    print("=" * 70)
    print("FINAL PAD-UFES TEST EVALUATION")
    print("=" * 70)

    targets, predictions = evaluate(
        model,
        loader,
        device,
    )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print(
        f"Test samples: {len(targets)}"
    )

    print(
        f"Accuracy:    {accuracy:.4f}"
    )

    print(
        f"Macro F1:    {macro_f1:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    print()
    print("CONFUSION MATRIX")

    print(
        "Rows = true class, "
        "Columns = predicted class"
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=np.arange(
            len(PAD_CLASSES)
        ),
    )

    print(
        "              "
        + " ".join(
            f"{name:>6}"
            for name in PAD_CLASSES
        )
    )

    for name, row in zip(
        PAD_CLASSES,
        matrix,
    ):
        print(
            f"{name:>8} "
            + " ".join(
                f"{value:6d}"
                for value in row
            )
        )

    print()
    print("PER-CLASS TEST METRICS")

    print(
        classification_report(
            targets,
            predictions,
            target_names=PAD_CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "No training was performed."
    )

    print(
        "Checkpoint was not modified."
    )


if __name__ == "__main__":
    main()
