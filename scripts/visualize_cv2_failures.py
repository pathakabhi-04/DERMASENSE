from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


ROOT = Path("/workspace/dermasense")

DEFAULT_METRICS = (
    ROOT / "evaluation/cv2/val_per_image_metrics.csv"
)

DEFAULT_OUTPUT = (
    ROOT / "evaluation/cv2/failure_cases"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize CV-2 detection failure cases."
    )

    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    return parser.parse_args()


def resolve_image(image_id: str) -> Path:
    path = (
        ROOT
        / "data/raw/itobos/_train/_train/images"
        / f"{image_id}.png"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    return path


def load_yolo_boxes(
    image_id: str,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    label_path = (
        ROOT
        / "data/raw/itobos/_train/_train/labels"
        / f"{image_id}.txt"
    )

    boxes = []

    with label_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            _, xc, yc, w, h = map(
                float,
                parts,
            )

            x1 = int(
                (xc - w / 2) * width
            )
            y1 = int(
                (yc - h / 2) * height
            )
            x2 = int(
                (xc + w / 2) * width
            )
            y2 = int(
                (yc + h / 2) * height
            )

            boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

    return boxes


def draw_boxes(
    image,
    boxes,
    label,
):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def make_panel(
    image_id: str,
    metrics_row: pd.Series,
) -> tuple:
    image_path = resolve_image(image_id)

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Unable to read image: {image_path}"
        )

    height, width = image.shape[:2]

    gt_boxes = load_yolo_boxes(
        image_id,
        width,
        height,
    )

    gt_panel = image.copy()
    draw_boxes(
        gt_panel,
        gt_boxes,
        "GT",
    )

    # The evaluator CSV does not contain prediction
    # coordinates, only aggregate prediction counts.
    #
    # Therefore this panel deliberately does not
    # fabricate predicted boxes.
    #
    # It instead provides the original image,
    # ground-truth boxes, and a metric summary.
    summary_panel = image.copy()

    lines = [
        f"Image: {image_id}",
        f"GT lesions: {int(metrics_row.gt_boxes)}",
        f"Predictions: {int(metrics_row.pred_boxes)}",
        f"Matched: {int(metrics_row.matched_boxes)}",
        f"False positives: {int(metrics_row.false_positives)}",
        f"Missed: {int(metrics_row.missed_boxes)}",
        f"Density: {metrics_row.density_bucket}",
    ]

    y = 35

    for line in lines:
        cv2.putText(
            summary_panel,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 35

    separator = 4

    return (
        cv2.hconcat(
            [
                gt_panel,
                cv2.copyMakeBorder(
                    summary_panel,
                    0,
                    0,
                    separator,
                    separator,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                ),
            ]
        )
    )


def main() -> None:
    args = parse_args()

    if not args.metrics.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {args.metrics}"
        )

    if args.top_k <= 0:
        raise ValueError(
            "top-k must be positive."
        )

    metrics = pd.read_csv(
        args.metrics
    )

    required = {
        "image_id",
        "gt_boxes",
        "pred_boxes",
        "matched_boxes",
        "false_positives",
        "missed_boxes",
        "density_bucket",
        "zero_lesion",
    }

    missing = required - set(
        metrics.columns
    )

    if missing:
        raise ValueError(
            f"Metrics CSV missing columns: "
            f"{sorted(missing)}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # 1. Worst recall cases
    # ------------------------------------------------------------------

    positive = metrics[
        metrics["gt_boxes"] > 0
    ].copy()

    positive["recall"] = (
        positive["matched_boxes"]
        / positive["gt_boxes"]
    )

    worst_recall = (
        positive
        .sort_values(
            [
                "recall",
                "gt_boxes",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(args.top_k)
    )

    # ------------------------------------------------------------------
    # 2. Worst false-positive cases
    # ------------------------------------------------------------------

    zero = metrics[
        metrics["zero_lesion"]
    ].copy()

    zero_fp = (
        zero[
            zero["pred_boxes"] > 0
        ]
        .sort_values(
            "pred_boxes",
            ascending=False,
        )
        .head(args.top_k)
    )

    # ------------------------------------------------------------------
    # 3. Dense-image failures
    # ------------------------------------------------------------------

    dense = positive[
        positive["density_bucket"] == "10+"
    ].copy()

    dense["recall"] = (
        dense["matched_boxes"]
        / dense["gt_boxes"]
    )

    dense_failures = (
        dense
        .sort_values(
            [
                "recall",
                "gt_boxes",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(args.top_k)
    )

    # ------------------------------------------------------------------
    # Save machine-readable failure tables
    # ------------------------------------------------------------------

    worst_recall.to_csv(
        args.output_dir
        / "worst_recall.csv",
        index=False,
    )

    zero_fp.to_csv(
        args.output_dir
        / "zero_lesion_false_positives.csv",
        index=False,
    )

    dense_failures.to_csv(
        args.output_dir
        / "dense_failures.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Generate image panels
    # ------------------------------------------------------------------

    groups = {
        "worst_recall": worst_recall,
        "zero_lesion_fp": zero_fp,
        "dense_failures": dense_failures,
    }

    for group_name, group in groups.items():
        group_dir = (
            args.output_dir
            / group_name
        )

        group_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for rank, (_, row) in enumerate(
            group.iterrows(),
            start=1,
        ):
            panel = make_panel(
                str(row["image_id"]),
                row,
            )

            output = (
                group_dir
                / f"{rank:02d}_{row['image_id']}.jpg"
            )

            cv2.imwrite(
                str(output),
                panel,
            )

    print("=" * 80)
    print("CV-2 FAILURE ANALYSIS")
    print("=" * 80)

    print()
    print(
        f"Worst recall cases: "
        f"{len(worst_recall)}"
    )

    print(
        f"Zero-lesion FP cases: "
        f"{len(zero_fp)}"
    )

    print(
        f"Dense failures: "
        f"{len(dense_failures)}"
    )

    print()
    print(
        f"Saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
