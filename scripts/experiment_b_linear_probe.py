from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch import nn
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

FEATURE_DIM = 2048


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment B: frozen ISIC ResNet-50 linear probe on PAD-UFES."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="ISIC experiment configuration.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Frozen ISIC ResNet-50 checkpoint.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum linear-probe epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
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
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def build_dataset(split: str):
    return CVDatasetTorch(
        dataset_id="pad_ufes",
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

def build_loader(dataset, batch_size: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_images_and_targets,
    )

def extract_features(
    backbone,
    loader,
    device,
):
    """
    Extract frozen backbone features.

    No gradients are created anywhere in this function.
    """

    backbone.eval()

    features = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            target = batch["target"]

            batch_features = backbone(images)

            features.append(
                batch_features.cpu()
            )

            targets.append(
                target.cpu()
            )

    return (
        torch.cat(features, dim=0),
        torch.cat(targets, dim=0),
    )


def make_class_weights(targets: torch.Tensor):
    """
    sqrt inverse-frequency weights.

    This prevents the very small MEL class from being
    completely dominated by the large BCC/ACK classes,
    while avoiding the instability of full inverse frequency.
    """

    counts = torch.bincount(
        targets,
        minlength=len(PAD_CLASSES),
    ).float()

    weights = torch.sqrt(
        counts.sum() / counts.clamp_min(1.0)
    )

    weights = weights / weights.mean()

    return weights


def evaluate_probe(
    classifier,
    features,
    targets,
    device,
):
    classifier.eval()

    with torch.no_grad():
        logits = classifier(
            features.to(device)
        )

    predictions = (
        logits.argmax(dim=1)
        .cpu()
        .numpy()
    )

    y_true = targets.numpy()

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "predictions": predictions,
    }


def main():
    args = parse_args()

    device = resolve_device(args.device)

    print("=" * 70)
    print("DERMASENSE EXPERIMENT B")
    print("FROZEN ISIC RESNET-50 → PAD-UFES LINEAR PROBE")
    print("=" * 70)

    print(f"Device:      {device}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"LR:          {args.learning_rate}")

    # ------------------------------------------------------------
    # Load frozen ISIC checkpoint
    # ------------------------------------------------------------

    config = NativeClassifierConfig(
        backbone="resnet50",
        pretrained=False,
        dropout=0.0,
    )

    model = DermaSenseNativeClassifier(config)

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
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
            "Could not find model state dict in checkpoint."
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)

    # ------------------------------------------------------------
    # Freeze backbone
    # ------------------------------------------------------------

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    model.backbone.eval()

    trainable_backbone = sum(
        p.numel()
        for p in model.backbone.parameters()
        if p.requires_grad
    )

    total_backbone = sum(
        p.numel()
        for p in model.backbone.parameters()
    )

    print()
    print("BACKBONE")
    print(f"Feature dimension: {model.backbone.feature_dim}")
    print(f"Backbone parameters: {total_backbone:,}")
    print(f"Trainable backbone parameters: {trainable_backbone:,}")

    if trainable_backbone != 0:
        raise RuntimeError(
            "Backbone is not completely frozen."
        )

    # ------------------------------------------------------------
    # PAD-UFES datasets
    # ------------------------------------------------------------

    print()
    print("DATASETS")

    train_dataset = build_dataset("train")
    val_dataset = build_dataset("val")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Classes:       {train_dataset.class_names}")

    if train_dataset.class_names != PAD_CLASSES:
        raise RuntimeError(
            "Unexpected PAD-UFES class ordering: "
            f"{train_dataset.class_names}"
        )

    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        shuffle=False,
    )

    val_loader = build_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
    )

    # ------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("FROZEN FEATURE EXTRACTION")
    print("=" * 70)

    train_features, train_targets = extract_features(
        model.backbone,
        train_loader,
        device,
    )

    val_features, val_targets = extract_features(
        model.backbone,
        val_loader,
        device,
    )

    print(
        f"Train features: {tuple(train_features.shape)}"
    )

    print(
        f"Val features:   {tuple(val_features.shape)}"
    )

    if train_features.shape[1] != FEATURE_DIM:
        raise RuntimeError(
            f"Expected {FEATURE_DIM}-D features, "
            f"got {train_features.shape[1]}."
        )

    # ------------------------------------------------------------
    # Linear probe
    # ------------------------------------------------------------

    classifier = nn.Linear(
        FEATURE_DIM,
        len(PAD_CLASSES),
    ).to(device)

    trainable_parameters = sum(
        p.numel()
        for p in classifier.parameters()
        if p.requires_grad
    )

    print()
    print("LINEAR PROBE")
    print(
        f"Classifier: {FEATURE_DIM} → {len(PAD_CLASSES)}"
    )
    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    class_weights = make_class_weights(
        train_targets
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    print(
        "Class weights:",
        class_weights.detach().cpu().numpy(),
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    print()
    print("=" * 70)
    print("LINEAR PROBE TRAINING")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):

        classifier.train()

        permutation = torch.randperm(
            len(train_features)
        )

        running_loss = 0.0
        sample_count = 0

        for start in range(
            0,
            len(permutation),
            args.batch_size,
        ):

            indices = permutation[
                start:start + args.batch_size
            ]

            batch_features = train_features[
                indices
            ].to(device)

            batch_targets = train_targets[
                indices
            ].to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = classifier(
                batch_features
            )

            loss = criterion(
                logits,
                batch_targets,
            )

            loss.backward()

            optimizer.step()

            batch_size_actual = len(indices)

            running_loss += (
                loss.item()
                * batch_size_actual
            )

            sample_count += batch_size_actual

        train_loss = (
            running_loss / sample_count
        )

        train_metrics = evaluate_probe(
            classifier,
            train_features,
            train_targets,
            device,
        )

        val_metrics = evaluate_probe(
            classifier,
            val_features,
            val_targets,
            device,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_macro_f1="
            f"{train_metrics['macro_f1']:.4f} | "
            f"val_macro_f1="
            f"{val_metrics['macro_f1']:.4f} | "
            f"val_accuracy="
            f"{val_metrics['accuracy']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics[
                "macro_f1"
            ]

            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in classifier.state_dict().items()
            }

            epochs_without_improvement = 0

            print(
                "  → saved best linear probe"
                f" (val_macro_f1={best_val_f1:.4f})"
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                f"  → early stopping after "
                f"{args.patience} epochs without improvement"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No best linear-probe checkpoint was produced."
        )

    classifier.load_state_dict(
        best_state
    )

    # ------------------------------------------------------------
    # Validation result
    # ------------------------------------------------------------

    best_val_metrics = evaluate_probe(
        classifier,
        val_features,
        val_targets,
        device,
    )

    print()
    print("=" * 70)
    print("BEST VALIDATION RESULT")
    print("=" * 70)

    print(f"Best epoch:        {best_epoch}")
    print(
        f"Best val Macro-F1: "
        f"{best_val_metrics['macro_f1']:.4f}"
    )
    print(
        f"Val Accuracy:      "
        f"{best_val_metrics['accuracy']:.4f}"
    )
    print(
        f"Val Weighted F1:   "
        f"{best_val_metrics['weighted_f1']:.4f}"
    )

    # ------------------------------------------------------------
    # FINAL TEST — first and only use of test set
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL PAD-UFES TEST EVALUATION")
    print("=" * 70)

    test_dataset = build_dataset("test")

    test_loader = build_loader(
        test_dataset,
        args.batch_size,
        shuffle=False,
    )

    test_features, test_targets = extract_features(
        model.backbone,
        test_loader,
        device,
    )

    test_metrics = evaluate_probe(
        classifier,
        test_features,
        test_targets,
        device,
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    print(
        f"Accuracy:    {test_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1:    {test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: {test_metrics['weighted_f1']:.4f}"
    )

    print()
    print("PER-CLASS TEST METRICS")

    print(
        classification_report(
            test_targets.numpy(),
            test_metrics["predictions"],
            target_names=PAD_CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    print("=" * 70)
    print("EXPERIMENT B COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
