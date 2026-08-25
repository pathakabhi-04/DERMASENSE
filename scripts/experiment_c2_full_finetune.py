from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)
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
        description=(
            "Experiment C2: full ResNet-50 fine-tuning "
            "from frozen ISIC checkpoint on PAD-UFES."
        )
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
        help="Maximum fine-tuning epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate for the full pretrained ResNet-50 backbone.",
    )

    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the PAD-UFES classification head.",
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


def build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_images_and_targets,
    )


def make_class_weights(
    targets: torch.Tensor,
):
    """
    Same sqrt inverse-frequency weighting used
    by Experiment B and C1.
    """

    counts = torch.bincount(
        targets,
        minlength=len(PAD_CLASSES),
    ).float()

    weights = torch.sqrt(
        counts.sum()
        / counts.clamp_min(1.0)
    )

    weights = weights / weights.mean()

    return weights


def evaluate(
    model,
    loader,
    device,
    criterion,
):
    model.eval()

    running_loss = 0.0
    sample_count = 0

    targets = []
    predictions = []

    with torch.no_grad():
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
                "pad_ufes",
            )

            loss = criterion(
                logits,
                batch_targets,
            )

            batch_size_actual = len(
                batch_targets
            )

            running_loss += (
                loss.item()
                * batch_size_actual
            )

            sample_count += (
                batch_size_actual
            )

            batch_predictions = (
                logits.argmax(dim=1)
                .cpu()
                .numpy()
            )

            predictions.append(
                batch_predictions
            )

            targets.append(
                batch_targets
                .cpu()
                .numpy()
            )

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)

    return {
        "loss": (
            running_loss
            / sample_count
        ),
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "targets": y_true,
        "predictions": y_pred,
    }


def save_best_checkpoint(
    path: Path,
    model,
    best_epoch: int,
    best_val_f1: float,
    checkpoint_path: Path,
    args,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            },
            "experiment": "C2",
            "architecture": (
                "resnet50_full_finetune"
            ),
            "dataset_id": "pad_ufes",
            "classes": PAD_CLASSES,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "backbone_learning_rate": (
                args.backbone_learning_rate
            ),
            "head_learning_rate": (
                args.head_learning_rate
            ),
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "source_checkpoint": str(
                checkpoint_path
            ),
        },
        path,
    )


def main():
    args = parse_args()

    device = resolve_device(
        args.device
    )

    print("=" * 70)
    print("DERMASENSE EXPERIMENT C2")
    print(
        "FULL RESNET-50 FINE-TUNING: "
        "ISIC → PAD-UFES"
    )
    print("=" * 70)

    print(
        f"Device:              {device}"
    )

    print(
        f"Checkpoint:          "
        f"{args.checkpoint}"
    )

    print(
        f"Epochs:              "
        f"{args.epochs}"
    )

    print(
        f"Batch size:          "
        f"{args.batch_size}"
    )

    print(
        f"Backbone LR:         "
        f"{args.backbone_learning_rate}"
    )

    print(
        f"Head LR:             "
        f"{args.head_learning_rate}"
    )

    print(
        f"Weight decay:        "
        f"{args.weight_decay}"
    )

    print(
        f"Early-stop patience: "
        f"{args.patience}"
    )

    # ------------------------------------------------------------
    # Load ISIC ResNet-50 checkpoint
    # ------------------------------------------------------------

    model_config = NativeClassifierConfig(
        backbone="resnet50",
        pretrained=False,
        dropout=0.0,
    )

    model = DermaSenseNativeClassifier(
        model_config
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint does not exist: "
            f"{checkpoint_path}"
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

    # ------------------------------------------------------------
    # C2 trainability policy
    #
    # Entire ResNet-50 backbone:
    #   TRAINABLE
    #
    # PAD-UFES head:
    #   TRAINABLE
    #
    # ISIC head:
    #   FROZEN
    # ------------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.backbone.parameters():
        parameter.requires_grad = True

    for parameter in (
        model.pad_ufes_head.parameters()
    ):
        parameter.requires_grad = True

    for parameter in (
        model.isic2019_head.parameters()
    ):
        parameter.requires_grad = False

    # ------------------------------------------------------------
    # Verify trainable components
    # ------------------------------------------------------------

    total_backbone = sum(
        p.numel()
        for p in model.backbone.parameters()
    )

    trainable_backbone = sum(
        p.numel()
        for p in model.backbone.parameters()
        if p.requires_grad
    )

    trainable_pad_head = sum(
        p.numel()
        for p in model.pad_ufes_head.parameters()
        if p.requires_grad
    )

    trainable_isic_head = sum(
        p.numel()
        for p in model.isic2019_head.parameters()
        if p.requires_grad
    )

    total_trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "TRAINABLE COMPONENTS"
    )

    print(
        f"Total backbone parameters:     "
        f"{total_backbone:,}"
    )

    print(
        f"Trainable backbone parameters: "
        f"{trainable_backbone:,}"
    )

    print(
        f"Trainable PAD head parameters:  "
        f"{trainable_pad_head:,}"
    )

    print(
        f"Trainable ISIC head parameters: "
        f"{trainable_isic_head:,}"
    )

    print(
        f"Total trainable parameters:     "
        f"{total_trainable:,}"
    )

    if trainable_backbone != total_backbone:
        raise RuntimeError(
            "C2 requires the entire "
            "ResNet-50 backbone to be trainable."
        )

    if trainable_isic_head != 0:
        raise RuntimeError(
            "ISIC head must remain frozen."
        )

    if trainable_pad_head == 0:
        raise RuntimeError(
            "PAD-UFES head must be trainable."
        )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    print()
    print("DATASETS")

    train_dataset = build_dataset(
        "train"
    )

    val_dataset = build_dataset(
        "val"
    )

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Val samples:   "
        f"{len(val_dataset)}"
    )

    print(
        f"Classes:       "
        f"{train_dataset.class_names}"
    )

    if (
        train_dataset.class_names
        != PAD_CLASSES
    ):
        raise RuntimeError(
            "Unexpected PAD-UFES class ordering: "
            f"{train_dataset.class_names}"
        )

    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
    )

    val_loader = build_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
    )

    # ------------------------------------------------------------
    # Class weighting
    # ------------------------------------------------------------

    train_targets = torch.tensor(
        [
            train_dataset[i]["target"]
            for i in range(
                len(train_dataset)
            )
        ],
        dtype=torch.long,
    )

    class_weights = make_class_weights(
        train_targets
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    print()
    print("CLASS WEIGHTS")

    print(
        class_weights.detach()
        .cpu()
        .numpy()
    )

    # ------------------------------------------------------------
    # Differential learning rates
    # ------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    model.backbone.parameters()
                ),
                "lr": (
                    args.backbone_learning_rate
                ),
            },
            {
                "params": (
                    model.pad_ufes_head.parameters()
                ),
                "lr": (
                    args.head_learning_rate
                ),
            },
        ],
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------
    # Best checkpoint tracking
    # ------------------------------------------------------------

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    best_checkpoint_path = Path(
        "checkpoints/"
        "pad_ufes_c2_full_finetune_best.pt"
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("C2 TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        # The ISIC head is frozen and is not used
        # by the PAD-UFES forward path.
        model.isic2019_head.eval()

        running_loss = 0.0
        sample_count = 0

        for batch in train_loader:

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            targets = batch["target"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                images,
                "pad_ufes",
            )

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()

            optimizer.step()

            batch_size_actual = len(
                targets
            )

            running_loss += (
                loss.item()
                * batch_size_actual
            )

            sample_count += (
                batch_size_actual
            )

        train_loss = (
            running_loss
            / sample_count
        )

        train_metrics = evaluate(
            model,
            train_loader,
            device,
            criterion,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        print(
            f"Epoch {epoch:03d}/"
            f"{args.epochs:03d} | "
            f"train_loss="
            f"{train_loss:.4f} | "
            f"train_macro_f1="
            f"{train_metrics['macro_f1']:.4f} | "
            f"val_loss="
            f"{val_metrics['loss']:.4f} | "
            f"val_macro_f1="
            f"{val_metrics['macro_f1']:.4f}"
        )

        if (
            val_metrics["macro_f1"]
            > best_val_f1
        ):

            best_val_f1 = (
                val_metrics["macro_f1"]
            )

            best_epoch = epoch

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

            save_best_checkpoint(
                path=best_checkpoint_path,
                model=model,
                best_epoch=best_epoch,
                best_val_f1=best_val_f1,
                checkpoint_path=checkpoint_path,
                args=args,
            )

            print(
                "  → saved best C2 checkpoint "
                f"(val_macro_f1="
                f"{best_val_f1:.4f})"
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                "  → early stopping after "
                f"{args.patience} epochs "
                "without improvement"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No best C2 checkpoint was produced."
        )

    # ------------------------------------------------------------
    # Restore best validation model
    # ------------------------------------------------------------

    model.load_state_dict(
        best_state,
        strict=True,
    )

    # ------------------------------------------------------------
    # Best validation result
    # ------------------------------------------------------------

    best_val_metrics = evaluate(
        model,
        val_loader,
        device,
        criterion,
    )

    print()
    print("=" * 70)
    print("BEST C2 VALIDATION RESULT")
    print("=" * 70)

    print(
        f"Best epoch:        "
        f"{best_epoch}"
    )

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

    print()
    print(
        "VALIDATION PER-CLASS METRICS"
    )

    print(
        classification_report(
            best_val_metrics["targets"],
            best_val_metrics["predictions"],
            target_names=PAD_CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    # ------------------------------------------------------------
    # FINAL TEST
    #
    # First and only use of test set.
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL PAD-UFES TEST EVALUATION")
    print("=" * 70)

    test_dataset = build_dataset(
        "test"
    )

    test_loader = build_loader(
        test_dataset,
        args.batch_size,
        shuffle=False,
    )

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        criterion,
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    print(
        f"Accuracy:    "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1:    "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    print()
    print("CONFUSION MATRIX")
    print(
        "Rows = true class, "
        "Columns = predicted class"
    )

    matrix = confusion_matrix(
        test_metrics["targets"],
        test_metrics["predictions"],
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

    # ------------------------------------------------------------
    # Per-class test metrics
    # ------------------------------------------------------------

    print()
    print(
        "PER-CLASS TEST METRICS"
    )

    print(
        classification_report(
            test_metrics["targets"],
            test_metrics["predictions"],
            target_names=PAD_CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    print()
    print(
        f"Best checkpoint: "
        f"{best_checkpoint_path}"
    )

    print("=" * 70)
    print("EXPERIMENT C2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()