from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
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

# Locked F1 design choice.
SUPCON_LAMBDA = 0.10
PROJECTION_DIM = 128
SUPCON_TEMPERATURE = 0.10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Experiment F1: C1 layer4 fine-tuning with "
            "all-class supervised contrastive representation loss."
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
        default=1e-5,
    )

    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=1e-4,
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training.",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Configure random generators and deterministic PyTorch behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(
        True
    )


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
    generator=None,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_images_and_targets,
        generator=generator,
    )


def make_class_weights(
    targets: torch.Tensor,
):
    """
    Same sqrt inverse-frequency weighting used by C1.
    """
    counts = torch.bincount(
        targets,
        minlength=len(PAD_CLASSES),
    ).float()

    if torch.any(counts <= 0):
        raise RuntimeError(
            "Every PAD-UFES class must have "
            "at least one training sample."
        )

    weights = 1.0 / torch.sqrt(counts)

    weights = (
        weights
        / weights.mean()
    )

    return weights


class ProjectionHead(nn.Module):
    """
    Projection head used only by the supervised
    contrastive auxiliary objective.

    The classifier continues to consume the original
    2048-dimensional backbone representation.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        projection_dim: int = PROJECTION_DIM,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                input_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                input_dim,
                projection_dim,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
    ):
        return self.network(features)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
):
    """
    All-class supervised contrastive loss.

    Each sample treats every other sample of the same
    class in the batch as a positive. Samples from
    different classes are negatives.

    Samples without another same-class example in the
    batch contribute zero to the loss.
    """

    if features.ndim != 2:
        raise ValueError(
            "Expected [B, D] features."
        )

    if labels.ndim != 1:
        raise ValueError(
            "Expected [B] labels."
        )

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "Feature/label batch sizes differ."
        )

    batch_size = features.shape[0]

    if batch_size < 2:
        return features.sum() * 0.0

    features = nn.functional.normalize(
        features,
        p=2,
        dim=1,
    )

    logits = torch.matmul(
        features,
        features.T,
    ) / temperature

    logits_mask = (
        ~torch.eye(
            batch_size,
            dtype=torch.bool,
            device=features.device,
        )
    )

    logits = logits.masked_fill(
        ~logits_mask,
        float("-inf"),
    )

    labels_equal = (
        labels[:, None]
        == labels[None, :]
    )

    positive_mask = (
        labels_equal
        & logits_mask
    )

    positive_counts = positive_mask.sum(
        dim=1
    )

    valid = positive_counts > 0

    if not torch.any(valid):
        return features.sum() * 0.0

    max_logits = torch.max(
        logits,
        dim=1,
        keepdim=True,
    ).values

    stable_logits = (
        logits - max_logits
    )

    exp_logits = torch.exp(
        stable_logits
    )

    exp_logits = exp_logits.masked_fill(
        ~logits_mask,
        0.0,
    )

    log_denominator = torch.log(
        exp_logits.sum(
            dim=1,
            keepdim=True,
        )
        + 1e-12
    )

    log_prob = (
        stable_logits
        - log_denominator
    )

    positive_log_prob = (
        log_prob * positive_mask.float()
    )

    mean_positive_log_prob = (
        positive_log_prob.sum(dim=1)
        / positive_counts.clamp_min(1)
    )

    loss = -mean_positive_log_prob[
        valid
    ].mean()

    return loss


def evaluate(
    model,
    loader,
    device,
    criterion,
    projection_head=None,
):
    model.eval()

    if projection_head is not None:
        projection_head.eval()

    total_loss = 0.0
    total_samples = 0

    targets = []
    predictions = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            batch_targets = batch[
                "target"
            ].to(
                device,
                non_blocking=True,
            )

            features = (
                model.forward_features(
                    images
                )
            )

            logits = (
                model.pad_ufes_head(
                    features
                )
            )

            loss = criterion(
                logits,
                batch_targets,
            )

            batch_size = (
                batch_targets.shape[0]
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            batch_predictions = (
                torch.argmax(
                    logits,
                    dim=1,
                )
            )

            targets.extend(
                batch_targets.cpu()
                .numpy()
                .tolist()
            )

            predictions.extend(
                batch_predictions.cpu()
                .numpy()
                .tolist()
            )

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    return {
        "loss": (
            total_loss
            / max(total_samples, 1)
        ),
        "accuracy": accuracy_score(
            targets,
            predictions,
        ),
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            targets,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "targets": targets,
        "predictions": predictions,
    }


def print_metrics(
    metrics,
):
    print(
        f"Loss:       "
        f"{metrics['loss']:.4f}"
    )

    print(
        f"Accuracy:   "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1:   "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:"
        f"{metrics['weighted_f1']:.4f}"
    )


def main():
    args = parse_args()

    set_seed(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    print("=" * 70)
    print("DERMASENSE EXPERIMENT F1")
    print("=" * 70)

    print(
        "SUPERVISED CONTRASTIVE "
        "REPRESENTATION ADAPTATION"
    )

    print()
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
        f"Seed:                "
        f"{args.seed}"
    )

    print(
        f"SupCon lambda:       "
        f"{SUPCON_LAMBDA}"
    )

    print(
        f"SupCon temperature:  "
        f"{SUPCON_TEMPERATURE}"
    )

    print(
        f"Projection dimension: "
        f"{PROJECTION_DIM}"
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
    # Model
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

    print()
    print("SOURCE CHECKPOINT")

    print(
        f"Experiment:          "
        f"{checkpoint.get('experiment')}"
    )

    print(
        f"Architecture:        "
        f"{checkpoint.get('architecture')}"
    )

    print(
        f"Source seed:         "
        f"{checkpoint.get('seed')}"
    )

    print(
        f"Source best epoch:   "
        f"{checkpoint.get('best_epoch')}"
    )

    print(
        f"Source val Macro-F1: "
        f"{checkpoint.get('best_val_macro_f1')}"
    )

    # ------------------------------------------------------------
    # Freeze everything first.
    # ------------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    # ------------------------------------------------------------
    # Unfreeze ONLY layer4.
    # ------------------------------------------------------------

    layer4 = None

    for module_name, module in (
        model.backbone.features.named_children()
    ):
        if module_name == "7":
            layer4 = module
            break

    if layer4 is None:
        raise RuntimeError(
            "Could not identify ResNet-50 layer4."
        )

    for parameter in layer4.parameters():
        parameter.requires_grad = True

    # ------------------------------------------------------------
    # Unfreeze ONLY PAD-UFES head.
    # ------------------------------------------------------------

    for parameter in (
        model.pad_ufes_head.parameters()
    ):
        parameter.requires_grad = True

    # ------------------------------------------------------------
    # Projection head.
    #
    # This is newly initialized and trainable.
    # It is NOT used by the classifier.
    # ------------------------------------------------------------

    projection_head = ProjectionHead().to(
        device
    )

    # ------------------------------------------------------------
    # Verify trainable components.
    # ------------------------------------------------------------

    trainable_layer4 = sum(
        p.numel()
        for p in layer4.parameters()
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

    trainable_projection = sum(
        p.numel()
        for p in projection_head.parameters()
        if p.requires_grad
    )

    print()
    print("TRAINABLE COMPONENTS")

    print(
        f"Layer4 parameters:       "
        f"{trainable_layer4:,}"
    )

    print(
        f"PAD head parameters:      "
        f"{trainable_pad_head:,}"
    )

    print(
        f"ISIC head parameters:     "
        f"{trainable_isic_head:,}"
    )

    print(
        f"Projection parameters:    "
        f"{trainable_projection:,}"
    )

    if trainable_isic_head != 0:
        raise RuntimeError(
            "ISIC head must remain frozen."
        )

    if trainable_layer4 == 0:
        raise RuntimeError(
            "Layer4 must be trainable."
        )

    if trainable_pad_head == 0:
        raise RuntimeError(
            "PAD-UFES head must be trainable."
        )

    # ------------------------------------------------------------
    # Datasets
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
        tuple(train_dataset.class_names)
        != PAD_CLASSES
    ):
        raise RuntimeError(
            "Unexpected PAD-UFES class ordering."
        )

    # ------------------------------------------------------------
    # Deterministic loaders.
    # ------------------------------------------------------------

    train_generator = torch.Generator()

    train_generator.manual_seed(
        args.seed
    )

    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        generator=train_generator,
    )

    val_loader = build_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
    )

    # ------------------------------------------------------------
    # C1 class-weighted classification loss.
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

    class_weights = (
        make_class_weights(
            train_targets
        )
        .to(device)
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    print()
    print("CLASS WEIGHTS")

    for name, weight in zip(
        PAD_CLASSES,
        class_weights.cpu().tolist(),
    ):
        print(
            f"{name:>4}: "
            f"{weight:.6f}"
        )

    # ------------------------------------------------------------
    # Optimizer.
    #
    # C1 optimizer settings are retained.
    # Projection head uses the head LR.
    # ------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params": layer4.parameters(),
                "lr": args.backbone_learning_rate,
            },
            {
                "params": model.pad_ufes_head.parameters(),
                "lr": args.head_learning_rate,
            },
            {
                "params": projection_head.parameters(),
                "lr": args.head_learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("F1 TRAINING")
    print("=" * 70)

    best_epoch = 0
    best_val_macro_f1 = -float("inf")
    best_state = None
    best_projection_state = None
    epochs_without_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()
        projection_head.train()

        layer4.train()

        running_total_loss = 0.0
        running_ce_loss = 0.0
        running_supcon_loss = 0.0

        total_samples = 0

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

            features = (
                model.forward_features(
                    images
                )
            )

            logits = (
                model.pad_ufes_head(
                    features
                )
            )

            ce_loss = criterion(
                logits,
                targets,
            )

            projected = projection_head(
                features
            )

            supcon_loss = (
                supervised_contrastive_loss(
                    projected,
                    targets,
                    SUPCON_TEMPERATURE,
                )
            )

            total_loss = (
                ce_loss
                + SUPCON_LAMBDA
                * supcon_loss
            )

            total_loss.backward()

            optimizer.step()

            batch_size = (
                targets.shape[0]
            )

            total_samples += batch_size

            running_total_loss += (
                total_loss.item()
                * batch_size
            )

            running_ce_loss += (
                ce_loss.item()
                * batch_size
            )

            running_supcon_loss += (
                supcon_loss.item()
                * batch_size
            )

        train_total_loss = (
            running_total_loss
            / max(total_samples, 1)
        )

        train_ce_loss = (
            running_ce_loss
            / max(total_samples, 1)
        )

        train_supcon_loss = (
            running_supcon_loss
            / max(total_samples, 1)
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"total={train_total_loss:.4f} | "
            f"CE={train_ce_loss:.4f} | "
            f"SupCon={train_supcon_loss:.4f} | "
            f"val_macro_f1="
            f"{val_metrics['macro_f1']:.4f} | "
            f"val_acc="
            f"{val_metrics['accuracy']:.4f}"
        )

        if (
            val_metrics["macro_f1"]
            > best_val_macro_f1
        ):
            best_val_macro_f1 = (
                val_metrics["macro_f1"]
            )

            best_epoch = epoch

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value in (
                    model.state_dict()
                    .items()
                )
            }

            best_projection_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value in (
                    projection_head.state_dict()
                    .items()
                )
            }

            epochs_without_improvement = 0

            print(
                "  → saved best F1 state "
                f"(epoch {epoch})"
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                "  → early stopping"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No best F1 checkpoint state "
            "was produced."
        )

    # ------------------------------------------------------------
    # Restore best validation state.
    # ------------------------------------------------------------

    model.load_state_dict(
        best_state,
        strict=True,
    )

    projection_head.load_state_dict(
        best_projection_state,
        strict=True,
    )

    # ------------------------------------------------------------
    # Best validation result.
    # ------------------------------------------------------------

    best_val_metrics = evaluate(
        model,
        val_loader,
        device,
        criterion,
    )

    print()
    print("=" * 70)
    print("BEST F1 VALIDATION RESULT")
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
    print("VALIDATION PER-CLASS METRICS")

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
    # Save best F1 checkpoint.
    # ------------------------------------------------------------

    f1_checkpoint_path = Path(
        "checkpoints/"
        "pad_ufes_f1_supcon_best.pt"
    )

    f1_checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "projection_state_dict": (
                best_projection_state
            ),
            "experiment": "F1",
            "architecture": (
                "resnet50_layer4_finetune"
                "_all_class_supcon"
            ),
            "dataset_id": "pad_ufes",
            "seed": args.seed,
            "classes": PAD_CLASSES,
            "best_epoch": best_epoch,
            "best_val_macro_f1": (
                best_val_metrics["macro_f1"]
            ),
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
            "supcon_lambda": SUPCON_LAMBDA,
            "supcon_temperature": (
                SUPCON_TEMPERATURE
            ),
            "projection_dim": PROJECTION_DIM,
            "source_checkpoint": str(
                checkpoint_path
            ),
            "source_experiment": "C1",
            "deterministic": True,
            "trainable_stages": (
                "layer4",
                "pad_ufes_head",
                "projection_head",
            ),
        },
        f1_checkpoint_path,
    )

    print()
    print(
        f"Best checkpoint: "
        f"{f1_checkpoint_path}"
    )

    # ------------------------------------------------------------
    # FINAL TEST
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

    print()
    print("PER-CLASS TEST METRICS")

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
        f"{f1_checkpoint_path}"
    )

    print("=" * 70)
    print("EXPERIMENT F1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
