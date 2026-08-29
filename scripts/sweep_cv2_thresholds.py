from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IOU_THRESHOLD = 0.50

DEFAULT_THRESHOLDS = [
    0.001,
    0.002,
    0.005,
    0.010,
    0.020,
    0.030,
    0.050,
    0.075,
    0.100,
    0.150,
    0.200,
    0.250,
    0.300,
    0.400,
    0.500,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep CV-2 confidence thresholds using saved predictions."
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path(
            "data/raw/itobos/_train/_train/labels"
        ),
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )

    return parser.parse_args()


def box_iou(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:

    if len(a) == 0 or len(b) == 0:
        return np.zeros(
            (len(a), len(b)),
            dtype=np.float32,
        )

    ax1, ay1, ax2, ay2 = a.T
    bx1, by1, bx2, by2 = b.T

    ix1 = np.maximum(
        ax1[:, None],
        bx1[None, :],
    )

    iy1 = np.maximum(
        ay1[:, None],
        by1[None, :],
    )

    ix2 = np.minimum(
        ax2[:, None],
        bx2[None, :],
    )

    iy2 = np.minimum(
        ay2[:, None],
        by2[None, :],
    )

    iw = np.maximum(
        0.0,
        ix2 - ix1,
    )

    ih = np.maximum(
        0.0,
        iy2 - iy1,
    )

    intersection = iw * ih

    area_a = (
        np.maximum(0.0, ax2 - ax1)
        * np.maximum(0.0, ay2 - ay1)
    )

    area_b = (
        np.maximum(0.0, bx2 - bx1)
        * np.maximum(0.0, by2 - by1)
    )

    union = (
        area_a[:, None]
        + area_b[None, :]
        - intersection
    )

    return intersection / np.maximum(
        union,
        1e-12,
    )


def greedy_match(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
) -> int:

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0

    ious = box_iou(
        gt_boxes,
        pred_boxes,
    )

    candidates = []

    for gt_idx in range(len(gt_boxes)):
        for pred_idx in range(len(pred_boxes)):

            iou = float(
                ious[gt_idx, pred_idx]
            )

            if iou >= IOU_THRESHOLD:
                candidates.append(
                    (
                        iou,
                        gt_idx,
                        pred_idx,
                    )
                )

    candidates.sort(
        reverse=True
    )

    matched_gt = set()
    matched_pred = set()

    for _, gt_idx, pred_idx in candidates:

        if gt_idx in matched_gt:
            continue

        if pred_idx in matched_pred:
            continue

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

    return len(matched_gt)


def load_ground_truth(
    label_dir: Path,
    image_id: str,
    image_width: int,
    image_height: int,
) -> np.ndarray:

    path = (
        label_dir
        / f"{image_id}.txt"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    boxes = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                raise ValueError(
                    f"Malformed label: {path}"
                )

            class_id = int(parts[0])

            if class_id != 0:
                raise ValueError(
                    f"Unexpected class {class_id}: {path}"
                )

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            x1 = (
                xc - w / 2.0
            ) * image_width

            y1 = (
                yc - h / 2.0
            ) * image_height

            x2 = (
                xc + w / 2.0
            ) * image_width

            y2 = (
                yc + h / 2.0
            ) * image_height

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

    if not boxes:
        return np.empty(
            (0, 4),
            dtype=np.float32,
        )

    return np.asarray(
        boxes,
        dtype=np.float32,
    )


def load_image_sizes(
    image_ids: list[str],
) -> dict[str, tuple[int, int]]:

    sizes = {}

    manifest_paths = [
        Path(
            "data/splits/itobos_detection/train.csv"
        ),
        Path(
            "data/splits/itobos_detection/val.csv"
        ),
        Path(
            "data/splits/itobos_detection/test.csv"
        ),
    ]

    manifests = []

    for path in manifest_paths:

        if path.exists():
            manifests.append(
                pd.read_csv(path)
            )

    manifest_df = pd.concat(
        manifests,
        ignore_index=True,
    )

    manifest_df = manifest_df.drop_duplicates(
        subset=["image_id"]
    )

    path_by_id = dict(
        zip(
            manifest_df["image_id"],
            manifest_df["image_path"],
        )
    )

    for image_id in image_ids:

        if image_id not in path_by_id:
            raise KeyError(
                f"{image_id} not found in detection manifests."
            )

        image_path = Path(
            path_by_id[image_id]
        )

        if not image_path.is_absolute():
            image_path = (
                Path("/workspace/dermasense")
                / image_path
            )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        height, width = image.shape[:2]

        sizes[image_id] = (
            width,
            height,
        )

    return sizes


def main() -> None:

    args = parse_args()

    df = pd.read_csv(
        args.predictions
    )

    required = {
        "image_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Prediction CSV missing columns: "
            f"{sorted(missing)}"
        )

    image_ids = sorted(
        df["image_id"]
        .unique()
    )

    image_sizes = load_image_sizes(
        image_ids
    )

    image_predictions = {}

    for image_id, group in df.groupby(
        "image_id",
        sort=False,
    ):

        predictions = []

        for row in group.itertuples():

            predictions.append(
                (
                    float(row.confidence),
                    np.asarray(
                        [
                            row.x1,
                            row.y1,
                            row.x2,
                            row.y2,
                        ],
                        dtype=np.float32,
                    ),
                )
            )

        image_predictions[
            image_id
        ] = predictions

    gt_cache = {}

    for image_id in image_ids:

        width, height = image_sizes[
            image_id
        ]

        gt_cache[
            image_id
        ] = load_ground_truth(
            args.label_dir,
            image_id,
            width,
            height,
        )

    total_gt = sum(
        len(boxes)
        for boxes in gt_cache.values()
    )

    zero_images = [
        image_id
        for image_id, boxes
        in gt_cache.items()
        if len(boxes) == 0
    ]

    dense_images = [
        image_id
        for image_id, boxes
        in gt_cache.items()
        if len(boxes) >= 10
    ]

    rows = []

    for threshold in sorted(
        set(args.thresholds)
    ):

        total_matched = 0
        total_predictions = 0

        zero_fp_images = 0

        dense_gt = 0
        dense_matched = 0

        for image_id in image_ids:

            gt_boxes = gt_cache[
                image_id
            ]

            predictions = [
                box
                for confidence, box
                in image_predictions[
                    image_id
                ]
                if confidence >= threshold
            ]

            if predictions:

                pred_boxes = np.asarray(
                    predictions,
                    dtype=np.float32,
                )

            else:

                pred_boxes = np.empty(
                    (0, 4),
                    dtype=np.float32,
                )

            matched = greedy_match(
                gt_boxes,
                pred_boxes,
            )

            total_matched += matched
            total_predictions += len(
                pred_boxes
            )

            if (
                len(gt_boxes) == 0
                and len(pred_boxes) > 0
            ):
                zero_fp_images += 1

            if len(gt_boxes) >= 10:

                dense_gt += len(
                    gt_boxes
                )

                dense_matched += matched

        lesion_recall = (
            total_matched / total_gt
            if total_gt
            else 0.0
        )

        zero_fpr = (
            zero_fp_images
            / len(zero_images)
            if zero_images
            else 0.0
        )

        dense_recall = (
            dense_matched / dense_gt
            if dense_gt
            else 0.0
        )

        rows.append(
            {
                "confidence": threshold,
                "predictions": total_predictions,
                "matched": total_matched,
                "missed": (
                    total_gt
                    - total_matched
                ),
                "lesion_recall": lesion_recall,
                "zero_lesion_fpr": zero_fpr,
                "dense_10plus_recall": dense_recall,
                "recall_pass": (
                    lesion_recall >= 0.95
                ),
                "zero_fpr_pass": (
                    zero_fpr <= 0.05
                ),
                "dense_recall_pass": (
                    dense_recall >= 0.90
                ),
                "overall_pass": (
                    lesion_recall >= 0.95
                    and zero_fpr <= 0.05
                    and dense_recall >= 0.90
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        args.output,
        index=False,
    )

    print()
    print("=" * 80)
    print("CV-2 CONFIDENCE THRESHOLD SWEEP")
    print("=" * 80)
    print()
    print(
        f"Predictions: {args.predictions}"
    )
    print(
        f"GT lesions:  {total_gt}"
    )
    print(
        f"Zero images: {len(zero_images)}"
    )
    print(
        f"Dense images: {len(dense_images)}"
    )
    print()
    print(
        result.to_string(
            index=False
        )
    )
    print()
    print(
        f"Saved to: {args.output}"
    )


if __name__ == "__main__":
    main()
