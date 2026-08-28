from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.segmentation.dataset import ISIC2018SegmentationDataset
from src.segmentation.metrics import segmentation_metrics
from src.segmentation.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the DermaSense CV-2 segmentation model."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/cv2/best.pt"),
    )

    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path(
            "data/splits/isic2018_task1/test.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/cv2"),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def main():
    args = parse_args()

    device = resolve_device(args.device)

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if not args.test_csv.exists():
        raise FileNotFoundError(
            f"Test split not found: {args.test_csv}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("DERMASENSE CV-2 TEST EVALUATION")
    print("=" * 80)
    print(f"Device:       {device}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Test split:   {args.test_csv}")
    print(f"Image size:   {args.image_size}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Threshold:    {args.threshold}")
    print(f"Output:       {args.output_dir}")
    print()

    dataset = ISIC2018SegmentationDataset(
        split_csv=args.test_csv,
        image_size=(
            args.image_size,
            args.image_size,
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model().to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    per_image = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            batch_size = images.shape[0]

            for i in range(batch_size):
                metrics = segmentation_metrics(
                    logits[i:i + 1],
                    masks[i:i + 1],
                    threshold=args.threshold,
                )

                per_image.append(
                    {
                        "image_id": batch["image_id"][i],
                        "dice": float(metrics["dice"]),
                        "iou": float(metrics["iou"]),
                    }
                )

    results = pd.DataFrame(per_image)

    if len(results) != len(dataset):
        raise RuntimeError(
            "Number of evaluation results does not "
            "match test dataset size."
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "test_split": str(args.test_csv),
        "num_images": len(results),
        "threshold": args.threshold,
        "dice_mean": float(results["dice"].mean()),
        "dice_std": float(results["dice"].std()),
        "iou_mean": float(results["iou"].mean()),
        "iou_std": float(results["iou"].std()),
        "dice_median": float(results["dice"].median()),
        "iou_median": float(results["iou"].median()),
        "checkpoint_epoch": checkpoint.get(
            "epoch"
        ),
        "best_val_dice": checkpoint.get(
            "best_val_dice"
        ),
    }

    results.to_csv(
        args.output_dir / "per_image_metrics.csv",
        index=False,
    )

    with open(
        args.output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("CV-2 TEST RESULTS")
    print("=" * 80)
    print(f"Images:       {summary['num_images']}")
    print(f"Dice:         {summary['dice_mean']:.4f}")
    print(f"Dice std:     {summary['dice_std']:.4f}")
    print(f"IoU:          {summary['iou_mean']:.4f}")
    print(f"IoU std:      {summary['iou_std']:.4f}")
    print(f"Median Dice:  {summary['dice_median']:.4f}")
    print(f"Median IoU:   {summary['iou_median']:.4f}")
    print(f"Checkpoint epoch: {summary['checkpoint_epoch']}")
    print()
    print("Saved:")
    print(
        f"  {args.output_dir / 'summary.json'}"
    )
    print(
        f"  {args.output_dir / 'per_image_metrics.csv'}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
