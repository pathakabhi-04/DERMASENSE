from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.segmentation.dataset import ISIC2018SegmentationDataset
from src.segmentation.losses import build_loss
from src.segmentation.model import build_model
from src.segmentation.training import fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DermaSense CV-2 lesion segmentation model."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/splits/isic2018_task1"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/cv2"),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="bce_dice",
        choices=("bce_dice", "dice", "bce"),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit training batches for a smoke test.",
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit validation batches for a smoke test.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is preferred for the baseline experiment.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested but is not available."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def build_loader(
    csv_path: Path,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ISIC2018SegmentationDataset(
        split_csv=csv_path,
        image_size=(image_size, image_size),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def limit_loader(
    loader: DataLoader,
    max_batches: int | None,
) -> DataLoader:
    """Return a loader limited to a fixed number of batches."""

    if max_batches is None:
        return loader

    if max_batches <= 0:
        raise ValueError(
            "max batch limits must be positive"
        )

    sample_count = min(
        len(loader.dataset),
        max_batches * loader.batch_size,
    )

    dataset = Subset(
        loader.dataset,
        range(sample_count),
    )

    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=False,
    )


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)

    device = resolve_device(args.device)

    train_csv = args.data_root / "train.csv"
    val_csv = args.data_root / "val.csv"
    test_csv = args.data_root / "test.csv"

    for path in (train_csv, val_csv, test_csv):
        if not path.exists():
            raise FileNotFoundError(
                f"Required split artifact not found: {path}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("DERMASENSE CV-2 TRAINING")
    print("=" * 80)
    print(f"Device:       {device}")
    print("Model:        UNet")
    print(f"Loss:         {args.loss}")
    print(f"Image size:   {args.image_size}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Epochs:       {args.epochs}")
    print(f"Learning rate:{args.learning_rate}")
    print(f"Seed:         {args.seed}")
    print(f"Train split:  {train_csv}")
    print(f"Val split:    {val_csv}")
    print(f"Test split:   {test_csv}")
    print(f"Output:       {args.output_dir}")
    print()

    train_loader = build_loader(
        train_csv,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    val_loader = build_loader(
        val_csv,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    train_loader = limit_loader(
        train_loader,
        args.max_train_batches,
    )

    val_loader = limit_loader(
        val_loader,
        args.max_val_batches,
    )

    # Construct the test loader so the artifact is verified to be readable,
    # but deliberately do not pass it to the training loop.
    test_loader = build_loader(
        test_csv,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")
    print()

    model = build_model().to(device)

    criterion = build_loss(args.loss)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=args.output_dir,
    )

    config = {
        "model": "UNet",
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "loss": args.loss,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": str(device),
        "data_root": str(args.data_root),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
    }

    with open(
        args.output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    with open(
        args.output_dir / "history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("CV-2 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
