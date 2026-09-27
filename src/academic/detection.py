"""CV-2 on ResNet-50: torchvision Faster R-CNN (FPN v2) on iToBoS.

docs/academic_resnet50_three_task_spec.md, Sections 3 and 3.1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.v2 import functional as TF

from src.academic.data import read_bundle

LONG_SIDE = 1280
MATCH_IOU = 0.50


def build_detector(*, pretrained: bool = True, score_thresh: float = 0.05) -> torch.nn.Module:
    """COCO-pretrained Faster R-CNN, 2 classes (background, lesion), long side 1280."""
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, min_size=LONG_SIDE, max_size=LONG_SIDE,
                                       box_score_thresh=score_thresh)
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
    return model


def read_yolo_boxes(path: Path, width: int, height: int) -> np.ndarray:
    """YOLO (class xc yc w h, normalised) -> xyxy pixels, as analyze_cv2_predictions.py."""
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5 or int(parts[0]) != 0:
            raise ValueError(f"malformed label in {path}: {line!r}")
        xc, yc, w, h = map(float, parts[1:])
        boxes.append([(xc - w / 2) * width, (yc - h / 2) * height, (xc + w / 2) * width, (yc + h / 2) * height])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


class ITobosDataset(Dataset):
    def __init__(self, root: Path, split: str, *, train: bool) -> None:
        self.root = Path(root)
        df = read_bundle(self.root)
        self.df = df[(df.source == "itobos") & (df.split == split)].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"no iToBoS rows for split {split!r}")
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        with Image.open(self.root / row["image"]) as im:
            image = TF.to_image(im.convert("RGB"))
        height, width = image.shape[-2:]
        boxes = torch.from_numpy(read_yolo_boxes(self.root / row["label"], width, height))
        if self.train and torch.rand(()) < 0.5:
            image = TF.horizontal_flip(image)
            boxes = boxes[:, [2, 1, 0, 3]]
            boxes[:, [0, 2]] = width - boxes[:, [0, 2]]
        target = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}
        return TF.to_dtype(image, torch.float32, scale=True), target, row["image_id"]


def collate(batch):
    images, targets, ids = zip(*batch)
    return list(images), list(targets), list(ids)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def match_predictions(gt: np.ndarray, pred: np.ndarray, conf: np.ndarray) -> dict[int, int]:
    """Greedy IoU-first matching at IoU >= 0.5; returns {pred_index: gt_index}.

    Same policy as match_predictions() in scripts/analyze_cv2_predictions.py on main.
    """
    if len(gt) == 0 or len(pred) == 0:
        return {}
    ious = box_iou(gt, pred)
    cands = [(float(ious[g, p]), float(conf[p]), g, p)
             for g in range(len(gt)) for p in range(len(pred)) if ious[g, p] >= MATCH_IOU]
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
    used_g, matched = set(), {}
    for _, _, g, p in cands:
        if g in used_g or p in matched:
            continue
        used_g.add(g)
        matched[p] = g
    return matched


def density_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 3:
        return "1-3"
    if count <= 9:
        return "4-9"
    return "10+"
