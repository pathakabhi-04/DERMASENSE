from __future__ import annotations

import argparse
from pathlib import Path

from src.detection.model import build_detector


ROOT = Path("/workspace/dermasense")

DEFAULT_DATA = ROOT / "configs/cv2_itobos.yaml"
DEFAULT_WEIGHTS = "yolo11n.pt"
DEFAULT_PROJECT = ROOT / "runs/cv2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the DermaSense CV-2 lesion detector."
    )

    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="YOLO pretrained checkpoint.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Ultralytics dataset YAML.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT,
    )

    parser.add_argument(
        "--name",
        default="baseline",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}"
        )

    args.project.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("DERMASENSE CV-2 BASELINE TRAINING")
    print("=" * 80)
    print(f"Weights:    {args.weights}")
    print(f"Dataset:    {args.data}")
    print(f"Epochs:     {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch:      {args.batch}")
    print(f"Device:     {args.device}")
    print(f"Seed:       {args.seed}")
    print(f"Workers:    {args.workers}")
    print(f"Output:     {args.project / args.name}")
    print()

    model = build_detector(
        weights=args.weights,
        pretrained=True,
    )

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
