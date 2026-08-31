"""
CV-1.5 router -- Stage 2 training (learned classifier).

Only reached because Stage 1 (the classical heuristic) failed the
pre-committed gate: 80.0% pre_framed / 62.0% wide_field vs. a >=90%
per-class bar (analysis/quality/cv1_5_router/result.md). Per
docs/cv1_5_router_spec.md, Stage 2 is a ResNet18 fine-tune on
PAD-UFES-20 (-> pre_framed) vs. iToBoS 2024 (-> wide_field) train
splits, evaluated against the SAME held-out set Stage 1 used
(analysis/quality/cv1_5_router/eval_set.csv, fixed at seed=42) and the
same per-class gate -- so the two stages are directly comparable.

Ground-truth caveat: labels are dataset identity (proxy), not per-image-
verified framing. See the spec before interpreting results.

Intended to run on a GPU (RunPod) -- a multi-epoch pass over iToBoS's
~6.8k train images is impractically slow on CPU (CV-3's single UNet
forward pass alone measured ~0.63s on the development machine's CPU).
Use --smoke-test to sanity-check the script locally on CPU first (tiny
subset, 1 epoch, no real training).

Usage:
    python -m scripts.train_cv1_5_router --device cuda
    python -m scripts.train_cv1_5_router --smoke-test --device cpu
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.transforms import (
    ImageTransformConfig,
    build_eval_transform,
    build_train_transform,
)
from src.routing.classifier import build_router_model
from src.routing.dataset import (
    CLASS_NAMES,
    FramingImageDataset,
    build_framing_split,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

PAD_UFES_TRAIN = REPO_ROOT / "data/splits/pad_ufes/train.csv"
PAD_UFES_VAL = REPO_ROOT / "data/splits/pad_ufes/val.csv"
ITOBOS_TRAIN = REPO_ROOT / "data/splits/itobos_detection/train.csv"
ITOBOS_VAL = REPO_ROOT / "data/splits/itobos_detection/val.csv"

# The exact held-out set Stage 1 was scored against -- reused so Stage 2
# is a direct, apples-to-apples comparison, not a re-sample.
HELD_OUT_EVAL_SET = REPO_ROOT / "analysis/quality/cv1_5_router/eval_set.csv"

CHECKPOINT_DIR = REPO_ROOT / "checkpoints/cv1_5_router"
RESULT_DIR = REPO_ROOT / "analysis/quality/cv1_5_router"
PER_CLASS_GATE = 0.90


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the CV-1.5 router Stage 2 classifier."
    )
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "1 epoch, tiny per-class subset -- verifies the script runs "
            "end to end, not a real training run."
        ),
    )
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    table: pd.DataFrame,
    transform,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    train: bool,
) -> DataLoader:
    dataset = FramingImageDataset(table, transform)

    sampler = None
    shuffle = train
    if train:
        # Correct for PAD-UFES (~1.6k) vs iToBoS (~6.8k) class imbalance
        # via inverse-frequency weighted sampling rather than a loss
        # reweighting -- keeps the loss function plain CrossEntropyLoss.
        counts = table["label"].value_counts()
        weight_per_class = {
            label: 1.0 / count for label, count in counts.items()
        }
        sample_weights = table["label"].map(weight_per_class).to_numpy().copy()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu()
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())

    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)

    per_class_acc = {}
    for index, name in enumerate(CLASS_NAMES):
        class_mask = labels_arr == index
        if class_mask.sum() == 0:
            per_class_acc[name] = float("nan")
            continue
        per_class_acc[name] = float(
            (preds_arr[class_mask] == index).mean()
        )

    balanced_acc = float(
        np.nanmean(list(per_class_acc.values()))
    )

    return {
        "per_class_accuracy": per_class_acc,
        "balanced_accuracy": balanced_acc,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def run_final_holdout_eval(
    model: nn.Module, device: torch.device
) -> tuple[dict, pd.DataFrame]:
    """
    Score the trained model against the fixed Stage-1 held-out set.

    Returns both the aggregate metrics and a per-image predictions table
    (same schema as Stage 1's analysis/quality/cv1_5_router/predictions.csv,
    for direct comparison) -- so the pushed result isn't just a summary
    number, it's auditable the same way Stage 1's was.
    """
    eval_table = pd.read_csv(HELD_OUT_EVAL_SET)
    transform = build_eval_transform(ImageTransformConfig())
    loader = build_loader(
        eval_table,
        transform,
        batch_size=32,
        num_workers=2,
        device=device,
        train=False,
    )

    model.eval()
    records = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        for image_path, label_index, pred_index in zip(
            batch["image_path"], batch["label"].tolist(), preds
        ):
            label = CLASS_NAMES[label_index]
            predicted = CLASS_NAMES[pred_index]
            records.append(
                {
                    "image_path": image_path,
                    "label": label,
                    "predicted": predicted,
                    "correct": label == predicted,
                }
            )

    predictions = pd.DataFrame(records)
    per_class_acc = predictions.groupby("label")["correct"].mean().to_dict()
    balanced_acc = float(np.mean(list(per_class_acc.values())))

    metrics = {
        "per_class_accuracy": per_class_acc,
        "balanced_accuracy": balanced_acc,
    }
    return metrics, predictions


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    seed_everything(args.seed)

    max_per_class = 20 if args.smoke_test else None
    epochs = 1 if args.smoke_test else args.epochs

    train_table = build_framing_split(
        PAD_UFES_TRAIN,
        ITOBOS_TRAIN,
        max_per_class=max_per_class,
        seed=args.seed,
    )
    val_table = build_framing_split(
        PAD_UFES_VAL,
        ITOBOS_VAL,
        max_per_class=(10 if args.smoke_test else None),
        seed=args.seed,
    )

    print(f"Device: {device}")
    print(
        f"Train: {len(train_table)} images "
        f"({train_table['label'].value_counts().to_dict()})"
    )
    print(
        f"Val:   {len(val_table)} images "
        f"({val_table['label'].value_counts().to_dict()})"
    )

    transform_config = ImageTransformConfig()
    train_transform = build_train_transform(transform_config)
    eval_transform = build_eval_transform(transform_config)

    train_loader = build_loader(
        train_table,
        train_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        train=True,
    )
    val_loader = build_loader(
        val_table,
        eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        train=False,
    )

    model = build_router_model(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / "best.pt"
    best_balanced_acc = -1.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_balanced_acc={val_metrics['balanced_accuracy']:.4f}  "
            f"val_per_class={val_metrics['per_class_accuracy']}"
        )

        if val_metrics["balanced_accuracy"] > best_balanced_acc:
            best_balanced_acc = val_metrics["balanced_accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_balanced_accuracy": best_balanced_acc,
                },
                best_path,
            )
            print(f"  -> new best checkpoint saved: {best_path}")

    print()
    print("Training complete. Loading best checkpoint for final held-out eval...")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    holdout_metrics, holdout_predictions = run_final_holdout_eval(model, device)
    per_class = holdout_metrics["per_class_accuracy"]
    pre_framed_acc = per_class["pre_framed"]
    wide_field_acc = per_class["wide_field"]
    passed = (
        pre_framed_acc >= PER_CLASS_GATE
        and wide_field_acc >= PER_CLASS_GATE
    )

    summary_lines = [
        "CV-1.5 Domain Router -- Stage 2 (Classifier) Result",
        "=" * 60,
        "",
        f"Checkpoint: {best_path} (epoch {checkpoint['epoch']}, "
        f"val_balanced_accuracy={checkpoint['val_balanced_accuracy']:.4f})",
        "",
        f"Held-out set: {HELD_OUT_EVAL_SET} (same set Stage 1 was scored on)",
        f"pre_framed accuracy: {pre_framed_acc:.3f}  (gate >= {PER_CLASS_GATE})",
        f"wide_field accuracy: {wide_field_acc:.3f}  (gate >= {PER_CLASS_GATE})",
        "",
        f"RESULT: {'PASS' if passed else 'FAIL'}",
    ]
    summary_text = "\n".join(summary_lines)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "stage2_summary.txt").write_text(summary_text + "\n")
    holdout_predictions.to_csv(
        RESULT_DIR / "stage2_predictions.csv", index=False
    )

    print()
    print(summary_text)
    print(f"\nSummary written to:     {RESULT_DIR / 'stage2_summary.txt'}")
    print(f"Predictions written to: {RESULT_DIR / 'stage2_predictions.csv'}")


if __name__ == "__main__":
    main()
