from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO


ROOT = Path("/workspace/dermasense")

SPLIT_DIR = ROOT / "data/splits/itobos_detection"
LABEL_DIR = ROOT / "data/raw/itobos/_train/_train/labels"

MATCH_IOU = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose CV-2 prediction confidence and "
            "candidate generation."
        )
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
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

    inter_x1 = np.maximum(
        ax1[:, None],
        bx1[None, :],
    )

    inter_y1 = np.maximum(
        ay1[:, None],
        by1[None, :],
    )

    inter_x2 = np.minimum(
        ax2[:, None],
        bx2[None, :],
    )

    inter_y2 = np.minimum(
        ay2[:, None],
        by2[None, :],
    )

    inter_w = np.maximum(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_h = np.maximum(
        0.0,
        inter_y2 - inter_y1,
    )

    inter = inter_w * inter_h

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
        - inter
    )

    return inter / np.maximum(
        union,
        1e-12,
    )


def yolo_to_xyxy(
    xc: float,
    yc: float,
    w: float,
    h: float,
    width: int,
    height: int,
) -> list[float]:
    return [
        (xc - w / 2.0) * width,
        (yc - h / 2.0) * height,
        (xc + w / 2.0) * width,
        (yc + h / 2.0) * height,
    ]


def read_ground_truth(
    label_path: Path,
    width: int,
    height: int,
) -> np.ndarray:
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
                raise ValueError(
                    f"Malformed label: {label_path}: {line!r}"
                )

            class_id = int(parts[0])

            if class_id != 0:
                raise ValueError(
                    f"Unexpected class {class_id}: "
                    f"{label_path}"
                )

            boxes.append(
                yolo_to_xyxy(
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    width,
                    height,
                )
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


def match_predictions(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_conf: np.ndarray,
) -> tuple[list[int], list[int]]:
    """
    Match predictions to GT boxes using the same locked
    IoU threshold as CV-2 evaluation.

    Returns:
        matched_prediction_indices,
        missed_ground_truth_indices
    """

    if len(gt_boxes) == 0:
        return [], []

    if len(pred_boxes) == 0:
        return [], list(range(len(gt_boxes)))

    ious = box_iou(
        gt_boxes,
        pred_boxes,
    )

    candidates = []

    for gt_idx in range(len(gt_boxes)):
        for pred_idx in range(len(pred_boxes)):
            value = float(
                ious[gt_idx, pred_idx]
            )

            if value >= MATCH_IOU:
                candidates.append(
                    (
                        value,
                        float(pred_conf[pred_idx]),
                        gt_idx,
                        pred_idx,
                    )
                )

    # Same IoU-first greedy policy as evaluate_cv2.py.
    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        ),
        reverse=True,
    )

    matched_gt = set()
    matched_pred = set()

    for _, _, gt_idx, pred_idx in candidates:
        if gt_idx in matched_gt:
            continue

        if pred_idx in matched_pred:
            continue

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

    return (
        sorted(matched_pred),
        sorted(
            set(range(len(gt_boxes)))
            - matched_gt
        ),
    )


def density_bucket(
    count: int,
) -> str:
    if count == 0:
        return "0"

    if count <= 3:
        return "1-3"

    if count <= 9:
        return "4-9"

    return "10+"


def main() -> None:
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.weights}"
        )

    manifest = (
        SPLIT_DIR
        / f"{args.split}.csv"
    )

    if not manifest.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest}"
        )

    df = pd.read_csv(manifest)

    model = YOLO(
        str(args.weights)
    )

    prediction_records = []
    image_records = []

    print("=" * 80)
    print("CV-2 PREDICTION DIAGNOSTIC")
    print("=" * 80)
    print()
    print(f"Weights:        {args.weights}")
    print(f"Split:          {args.split}")
    print(f"Confidence:     {args.conf}")
    print(f"NMS IoU:        {args.iou}")
    print(f"Match IoU:      {MATCH_IOU}")
    print(f"Image size:     {args.imgsz}")

    for index, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):
        image_path = Path(
            row.image_path
        )

        if not image_path.is_absolute():
            image_path = ROOT / image_path

        label_path = (
            LABEL_DIR
            / f"{row.image_id}.txt"
        )

        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        result = results[0]

        image_height, image_width = (
            result.orig_shape
        )

        gt_boxes = read_ground_truth(
            label_path,
            image_width,
            image_height,
        )

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            pred_boxes = np.empty(
                (0, 4),
                dtype=np.float32,
            )

            pred_conf = np.empty(
                (0,),
                dtype=np.float32,
            )

        else:
            pred_boxes = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            pred_conf = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

        matched_pred, missed_gt = (
            match_predictions(
                gt_boxes,
                pred_boxes,
                pred_conf,
            )
        )

        matched_pred_set = set(
            matched_pred
        )

        gt_count = len(gt_boxes)

        # Record every prediction.
        for pred_idx in range(
            len(pred_boxes)
        ):
            prediction_records.append(
                {
                    "image_id": row.image_id,
                    "density_bucket": density_bucket(
                        gt_count
                    ),
                    "gt_boxes": gt_count,
                    "prediction_index": pred_idx,
                    "confidence": float(
                        pred_conf[pred_idx]
                    ),
                    "x1": float(
                        pred_boxes[pred_idx][0]
                    ),
                    "y1": float(
                        pred_boxes[pred_idx][1]
                    ),
                    "x2": float(
                        pred_boxes[pred_idx][2]
                    ),
                    "y2": float(
                        pred_boxes[pred_idx][3]
                    ),
                    "matched": (
                        pred_idx
                        in matched_pred_set
                    ),
                    "zero_lesion": (
                        gt_count == 0
                    ),
                }
            )

        image_records.append(
            {
                "image_id": row.image_id,
                "gt_boxes": gt_count,
                "pred_boxes": len(pred_boxes),
                "matched_boxes": len(
                    matched_pred
                ),
                "missed_boxes": len(
                    missed_gt
                ),
                "density_bucket": density_bucket(
                    gt_count
                ),
                "zero_lesion": (
                    gt_count == 0
                ),
            }
        )

        if index % 250 == 0:
            print(
                f"Processed {index}/{len(df)} images"
            )

    predictions = pd.DataFrame(
        prediction_records
    )

    images = pd.DataFrame(
        image_records
    )

    output_dir = (
        args.output
        if args.output is not None
        else (
            ROOT
            / "evaluation"
            / "cv2"
            / "prediction_diagnostics"
            / args.split
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_dir
        / "predictions.csv"
    )

    images_path = (
        output_dir
        / "per_image.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    images.to_csv(
        images_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    positive = images[
        images["gt_boxes"] > 0
    ]

    dense = images[
        images["density_bucket"] == "10+"
    ]

    zero = images[
        images["zero_lesion"]
    ]

    print()
    print("-" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("-" * 80)

    print()
    print(
        f"Images:              {len(images)}"
    )

    print(
        f"Ground-truth boxes:  "
        f"{int(images['gt_boxes'].sum())}"
    )

    print(
        f"Predictions:         "
        f"{int(images['pred_boxes'].sum())}"
    )

    print(
        f"Matched predictions: "
        f"{int(images['matched_boxes'].sum())}"
    )

    print(
        f"Missed GT boxes:     "
        f"{int(images['missed_boxes'].sum())}"
    )

    if len(positive):
        print(
            "Positive-image recall: "
            f"{positive['matched_boxes'].sum() / positive['gt_boxes'].sum():.4f}"
        )

    if len(dense):
        print(
            "10+ recall:             "
            f"{dense['matched_boxes'].sum() / dense['gt_boxes'].sum():.4f}"
        )

    if len(zero):
        print(
            "Zero-image FPR:         "
            f"{(zero['pred_boxes'] > 0).mean():.4f}"
        )

    if len(predictions):
        print()
        print("PREDICTION CONFIDENCE")

        print(
            f"min:                    "
            f"{predictions['confidence'].min():.4f}"
        )

        print(
            f"median:                 "
            f"{predictions['confidence'].median():.4f}"
        )

        print(
            f"p10:                    "
            f"{predictions['confidence'].quantile(0.10):.4f}"
        )

        print(
            f"p25:                    "
            f"{predictions['confidence'].quantile(0.25):.4f}"
        )

        print(
            f"p75:                    "
            f"{predictions['confidence'].quantile(0.75):.4f}"
        )

        print(
            f"p90:                    "
            f"{predictions['confidence'].quantile(0.90):.4f}"
        )

        print(
            f"max:                    "
            f"{predictions['confidence'].max():.4f}"
        )

        matched = predictions[
            predictions["matched"]
        ]

        unmatched = predictions[
            ~predictions["matched"]
        ]

        print()
        print(
            f"Matched prediction count: "
            f"{len(matched)}"
        )

        print(
            f"Unmatched prediction count: "
            f"{len(unmatched)}"
        )

        if len(matched):
            print(
                f"Matched confidence median: "
                f"{matched['confidence'].median():.4f}"
            )

            print(
                f"Matched confidence p10:     "
                f"{matched['confidence'].quantile(0.10):.4f}"
            )

        if len(unmatched):
            print(
                f"Unmatched confidence median: "
                f"{unmatched['confidence'].median():.4f}"
            )

    print()
    print(
        f"Predictions saved to: {predictions_path}"
    )

    print(
        f"Per-image metrics saved to: {images_path}"
    )


if __name__ == "__main__":
    main()
