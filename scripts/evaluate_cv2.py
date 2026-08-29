from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO


ROOT = Path("/workspace/dermasense")
SPLIT_DIR = ROOT / "data/splits/itobos_detection"
LABEL_DIR = ROOT / "data/raw/itobos/_train/_train/labels"

IOU_THRESHOLD = 0.50
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.70

RECALL_TARGET = 0.95
ZERO_LESION_FPR_TARGET = 0.05
DENSE_RECALL_TARGET = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CV-2 against locked acceptance metrics."
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=NMS_IOU_THRESHOLD,
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU for xyxy boxes.

    a: [N, 4]
    b: [M, 4]
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a.T
    bx1, by1, bx2, by2 = b.T

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)

    inter = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-12)


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height
    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height

    return [
        x1,
        y1,
        x2,
        y2,
    ]


def read_ground_truth(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    boxes = []

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Malformed label: {label_path}: {line!r}"
                )

            class_id = int(parts[0])

            if class_id != 0:
                raise ValueError(
                    f"Unexpected class {class_id} in {label_path}"
                )

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            boxes.append(
                yolo_to_xyxy(
                    x_center,
                    y_center,
                    width,
                    height,
                    image_width,
                    image_height,
                )
            )

    if not boxes:
        return np.empty((0, 4), dtype=np.float32)

    return np.asarray(boxes, dtype=np.float32)


def greedy_match(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
) -> tuple[int, int, int]:
    """
    Greedy one-to-one matching at the locked IoU threshold.

    Returns:
        matched_ground_truth,
        false_positives,
        missed_ground_truth
    """
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    ious = box_iou(gt_boxes, pred_boxes)

    candidates = []

    for gt_idx in range(len(gt_boxes)):
        for pred_idx in range(len(pred_boxes)):
            iou = float(ious[gt_idx, pred_idx])

            if iou >= IOU_THRESHOLD:
                candidates.append(
                    (iou, gt_idx, pred_idx)
                )

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()

    for _, gt_idx, pred_idx in candidates:
        if gt_idx in matched_gt:
            continue

        if pred_idx in matched_pred:
            continue

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

    matched = len(matched_gt)
    false_positives = len(pred_boxes) - len(matched_pred)
    missed = len(gt_boxes) - matched

    return matched, false_positives, missed


def density_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 3:
        return "1-3"
    if count <= 9:
        return "4-9"
    return "10+"


def main() -> None:
    args = parse_args()

    weights = args.weights

    if not weights.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights}"
        )

    manifest = SPLIT_DIR / f"{args.split}.csv"

    if not manifest.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest}"
        )

    df = pd.read_csv(manifest)

    model = YOLO(str(weights))

    records = []

    for index, row in enumerate(df.itertuples(index=False), start=1):
        image_path = Path(row.image_path)

        if not image_path.is_absolute():
            image_path = ROOT / image_path

        label_path = LABEL_DIR / f"{row.image_id}.txt"

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        result = results[0]

        image_height, image_width = result.orig_shape

        gt_boxes = read_ground_truth(
            label_path,
            image_width,
            image_height,
        )

        if result.boxes is None or len(result.boxes) == 0:
            pred_boxes = np.empty(
                (0, 4),
                dtype=np.float32,
            )
        else:
            pred_boxes = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

        matched, false_positives, missed = greedy_match(
            gt_boxes,
            pred_boxes,
        )

        gt_count = len(gt_boxes)

        records.append(
            {
                "image_id": row.image_id,
                "gt_boxes": gt_count,
                "pred_boxes": len(pred_boxes),
                "matched_boxes": matched,
                "false_positives": false_positives,
                "missed_boxes": missed,
                "density_bucket": density_bucket(gt_count),
                "zero_lesion": gt_count == 0,
            }
        )

        if index % 250 == 0:
            print(
                f"Processed {index}/{len(df)} images"
            )

    result_df = pd.DataFrame(records)

    total_gt = int(result_df["gt_boxes"].sum())
    total_matched = int(result_df["matched_boxes"].sum())

    lesion_recall = (
        total_matched / total_gt
        if total_gt
        else 0.0
    )

    zero_df = result_df[
        result_df["zero_lesion"]
    ]

    zero_fp_images = int(
        (
            zero_df["pred_boxes"] > 0
        ).sum()
    )

    zero_fpr = (
        zero_fp_images / len(zero_df)
        if len(zero_df)
        else 0.0
    )

    dense_df = result_df[
        result_df["density_bucket"] == "10+"
    ]

    dense_gt = int(
        dense_df["gt_boxes"].sum()
    )

    dense_matched = int(
        dense_df["matched_boxes"].sum()
    )

    dense_recall = (
        dense_matched / dense_gt
        if dense_gt
        else 0.0
    )

    print()
    print("=" * 80)
    print("DERMASENSE CV-2 ACCEPTANCE EVALUATION")
    print("=" * 80)

    print()
    print(f"Split:                 {args.split}")
    print(f"Images:                {len(result_df)}")
    print(f"Ground-truth lesions:  {total_gt}")
    print(f"Confidence threshold:  {args.conf}")
    print(f"NMS IoU threshold:     {args.iou}")
    print(f"Matching IoU threshold: {IOU_THRESHOLD}")

    print()
    print("LOCKED ACCEPTANCE METRICS")
    print("-" * 80)

    print(
        f"Lesion recall:          {lesion_recall:.4f} "
        f"(target >= {RECALL_TARGET:.2f})"
    )

    print(
        f"Zero-lesion image FPR:  {zero_fpr:.4f} "
        f"(target <= {ZERO_LESION_FPR_TARGET:.2f})"
    )

    print(
        f"10+ lesion recall:      {dense_recall:.4f} "
        f"(target >= {DENSE_RECALL_TARGET:.2f})"
    )

    recall_pass = lesion_recall >= RECALL_TARGET
    zero_pass = zero_fpr <= ZERO_LESION_FPR_TARGET
    dense_pass = dense_recall >= DENSE_RECALL_TARGET

    print()
    print("GATES")
    print("-" * 80)

    print(
        f"Lesion recall:          "
        f"{'PASS' if recall_pass else 'FAIL'}"
    )

    print(
        f"Zero-lesion FPR:        "
        f"{'PASS' if zero_pass else 'FAIL'}"
    )

    print(
        f"10+ lesion recall:      "
        f"{'PASS' if dense_pass else 'FAIL'}"
    )

    print()
    print(
        "OVERALL ACCEPTANCE: "
        f"{'PASS' if all((recall_pass, zero_pass, dense_pass)) else 'FAIL'}"
    )

    if args.output is None:
        output = (
            ROOT
            / "evaluation"
            / "cv2"
            / f"{args.split}_per_image_metrics.csv"
        )
    else:
        output = args.output

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        output,
        index=False,
    )

    print()
    print(f"Per-image metrics saved to: {output}")


if __name__ == "__main__":
    main()
