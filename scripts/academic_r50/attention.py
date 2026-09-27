"""Attention (Grad-CAM) maps for one task, scored against ground truth.

docs/academic_resnet50_three_task_spec.md, Section 5. Writes to <out-dir>/<task>/:
per_item.csv, sanity.csv, metrics.json and attention_<task>.jpg (contact sheet).

    PYTHONPATH=. python3 scripts/academic_r50/attention.py --task cv4 \
        --r50-root /workspace/dermasense_academic/data/academic_r50_bundle \
        --ham-root /workspace/dermasense_academic/data/academic_joint_bundle \
        --run-root /workspace/dermasense_academic/runs/academic_r50
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from src.academic import cam as C
from src.academic.data import IMAGENET_MEAN, IMAGENET_STD, ISIC2019_CLASSES, SegTestDataset, load_mask, read_bundle
from src.academic.detection import build_detector, match_predictions, read_yolo_boxes
from src.academic.metrics import macro_f1
from src.academic.model import AcademicModel, load_cv4_resnet50

SANITY_N = 50
SANITY_MAX_MEDIAN = 0.5
CLAIM_POINTING = 0.80
CLAIM_ENERGY = 0.50
SHEET_SEED = 42
CV2_CONF = 0.25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=("cv2", "cv3", "cv4"), required=True)
    p.add_argument("--r50-root", type=Path, required=True)
    p.add_argument("--ham-root", type=Path, help="academic_joint_bundle (cv4 only)")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="Smoke tests only.")
    return p.parse_args()


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sanity_indices(n_items: int) -> list[int]:
    rng = np.random.RandomState(42)
    return sorted(rng.choice(n_items, size=min(SANITY_N, n_items), replace=False).tolist())


def contact_sheet(panels: list[tuple[np.ndarray, str]], path: Path, cols: int = 5, size: int = 256) -> None:
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * (size + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (rgb, caption) in enumerate(panels):
        r, c = divmod(k, cols)
        sheet.paste(Image.fromarray(rgb).resize((size, size), Image.BILINEAR), (c * size, r * (size + 18)))
        draw.text((c * size + 4, r * (size + 18) + size + 2), caption, fill="black")
    sheet.save(path, quality=90)


def finish(out: Path, task: str, items: pd.DataFrame, sanity: pd.DataFrame, extra: dict) -> None:
    items.to_csv(out / "per_item.csv", index=False)
    sanity.to_csv(out / "sanity.csv", index=False)
    median_rho = float(sanity["spearman"].median())
    metrics = {
        "task": task,
        "n": int(len(items)),
        "pointing_game": float(items["pointing"].mean()),
        "energy_in_lesion": float(items["energy"].mean()),
        **({"iou_at_0.5": float(items["iou"].mean())} if "iou" in items else {}),
        "sanity_n": int(len(sanity)),
        "sanity_median_spearman": median_rho,
        "sanity_pass": median_rho < SANITY_MAX_MEDIAN,
        **extra,
    }
    metrics["claim_allowed"] = bool(metrics["sanity_pass"] and metrics["pointing_game"] >= CLAIM_POINTING
                                    and metrics["energy_in_lesion"] >= CLAIM_ENERGY)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


# --------------------------------------------------------------------- CV-4

def run_cv4(args, out: Path) -> None:
    dev = device()
    net = load_cv4_resnet50(args.r50_root / "cv4/isic2019_resnet50_weighted_best.pt").to(dev).eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    df = read_bundle(args.ham_root)
    df = df[(df.source == "ham") & (df.split == "test")].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    def load(i):
        row = df.iloc[i]
        with Image.open(args.ham_root / row["image"]) as im:
            im = im.convert("RGB")
            x = tf(im)[None].to(dev)
            rgb = np.array(im.resize((224, 224), Image.BILINEAR))
        gt = np.array(load_mask(args.ham_root / row["mask"]).resize((224, 224), Image.NEAREST)) > 127
        return x, rgb, gt

    rows, cams = [], {}
    for i in range(len(df)):
        x, _, gt = load(i)
        cam, pred = C.cam_classifier(net, x)
        up = C.upsample(cam, (224, 224))
        true = ISIC2019_CLASSES.index(df.label[i])
        rows.append({"image_id": df.image_id[i], "true": df.label[i], "pred": ISIC2019_CLASSES[pred],
                     "correct": pred == true, "pointing": C.pointing_hit(up, gt),
                     "energy": C.energy_fraction(up, gt), "iou": C.iou_at_half(up, gt)})
        cams[i] = (cam, pred)
    items = pd.DataFrame(rows)

    rnd = copy.deepcopy(net)
    C.randomize(rnd.layer4)
    sanity = []
    for i in sanity_indices(len(df)):
        x, _, _ = load(i)
        cam_r, _ = C.cam_classifier(rnd, x, target=cams[i][1])
        sanity.append({"image_id": df.image_id[i], "spearman": C.spearman(cams[i][0], cam_r)})

    panels = []
    for correct in (True, False):
        pick = items[items.correct == correct].sample(n=min(10, int((items.correct == correct).sum())),
                                                      random_state=SHEET_SEED)
        for i in pick.index:
            _, rgb, gt = load(i)
            img = C.draw_outline(C.overlay(rgb, C.upsample(cams[i][0], (224, 224))), gt)
            panels.append((img, f"{items.true[i]}->{items.pred[i]} {'ok' if correct else 'WRONG'}"))
    contact_sheet(panels, out / "attention_cv4.jpg")

    y = items.true.map(ISIC2019_CLASSES.index).to_numpy()
    p = items.pred.map(ISIC2019_CLASSES.index).to_numpy()
    finish(out, "cv4", items, pd.DataFrame(sanity),
           {"eval_set": "HAM10000 in ISIC 2019 test", "ham_test_macro_f1": macro_f1(y, p, 8),
            "method": "Grad-CAM, predicted-class logit, layer4 (7x7 at 224)"})


# --------------------------------------------------------------------- CV-3

def run_cv3(args, out: Path) -> None:
    dev = device()
    ck = torch.load(args.run_root / "S_seed42/best.pt", map_location="cpu", weights_only=False)
    model = AcademicModel("S", pretrained=False, seed=42)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    ds = SegTestDataset(args.r50_root, image_size=512, split="test")
    n = min(len(ds), args.limit or len(ds))

    def load(i):
        item = ds[i]
        with Image.open(args.r50_root / ds.df.iloc[i]["image"]) as im:
            rgb = np.array(im.convert("RGB"))
        return item["image"][None].to(dev), rgb, item["mask"][0].numpy() > 0.5

    rows, cams, empty = [], {}, []
    for i in range(n):
        x, _, gt = load(i)
        cam, region = C.cam_segmenter(model, x)
        if cam is None:
            empty.append(ds.df.image_id[i])
            continue
        up = C.upsample(cam, (512, 512))
        rows.append({"image_id": ds.df.image_id[i], "pointing": C.pointing_hit(up, gt),
                     "energy": C.energy_fraction(up, gt), "iou": C.iou_at_half(up, gt)})
        cams[i] = (cam, region)
    items = pd.DataFrame(rows)

    rnd = copy.deepcopy(model)
    C.randomize(rnd.encoder.layer4)
    sanity = []
    valid = sorted(cams)
    for k in sanity_indices(len(valid)):
        i = valid[k]
        x, _, _ = load(i)
        cam_r, _ = C.cam_segmenter(rnd, x, region=cams[i][1])
        sanity.append({"image_id": ds.df.image_id[i],
                       "spearman": 0.0 if cam_r is None else C.spearman(cams[i][0], cam_r)})

    panels = []
    for k in np.random.RandomState(SHEET_SEED).choice(len(valid), size=min(20, len(valid)), replace=False):
        i = valid[k]
        _, rgb, gt = load(i)
        img = C.draw_outline(C.overlay(rgb, C.upsample(cams[i][0], (512, 512))), gt)
        img = C.draw_outline(img, cams[i][1][0, 0].cpu().numpy(), color=(255, 255, 255))
        panels.append((img, str(ds.df.image_id[i])))
    contact_sheet(panels, out / "attention_cv3.jpg")

    finish(out, "cv3", items, pd.DataFrame(sanity),
           {"eval_set": "ISIC 2018 Task 1 test", "empty_prediction_excluded": len(empty),
            "empty_prediction_ids": empty,
            "method": "Seg-Grad-CAM, sum of mask logits in predicted mask, encoder layer4 (16x16 at 512)"})


# --------------------------------------------------------------------- CV-2

def run_cv2(args, out: Path) -> None:
    dev = device()
    run = args.run_root / "cv2"
    det = build_detector(pretrained=False)
    det.load_state_dict(torch.load(run / "final.pt", map_location="cpu", weights_only=False)["model"])
    det.to(dev).eval()
    bundle = read_bundle(args.r50_root).set_index(["source", "split", "image_id"])
    preds = pd.read_csv(run / "predictions.csv")

    def load(image_id):
        row = bundle.loc[("itobos", "val", image_id)]
        with Image.open(args.r50_root / row["image"]) as im:
            rgb = np.array(im.convert("RGB"))
        gt = read_yolo_boxes(args.r50_root / row["label"], rgb.shape[1], rgb.shape[0])
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(dev)
        return x, rgb, gt

    # Recover which GT box each kept true positive matched (same matching as the export).
    boxes = []
    for image_id, g in preds.groupby("image_id", sort=True):
        g = g.sort_values("prediction_index")
        if not ((g.confidence >= CV2_CONF) & g.matched).any():
            continue
        row = bundle.loc[("itobos", "val", image_id)]
        with Image.open(args.r50_root / row["image"]) as im:
            w, h = im.size
        gt = read_yolo_boxes(args.r50_root / row["label"], w, h)
        pb = g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        m = match_predictions(gt, pb, g.confidence.to_numpy())
        for k, r in enumerate(g.itertuples()):
            if r.confidence >= CV2_CONF and k in m:
                boxes.append({"image_id": image_id, "box": pb[k], "gt": gt[m[k]], "all_gt": gt,
                              "confidence": r.confidence})
    if args.limit:
        boxes = boxes[:args.limit]

    rows, cams = [], {}
    for j, b in enumerate(boxes):
        x, _, _ = load(b["image_id"])
        full, native = C.cam_detector_box(det, x, torch.from_numpy(b["box"]).to(dev), return_native=True)
        gt_mask = C.box_mask(full.shape, b["gt"])
        rows.append({"image_id": b["image_id"], "confidence": b["confidence"],
                     "box": [round(float(v), 1) for v in b["box"]],
                     "gt_box": [round(float(v), 1) for v in b["gt"]],
                     "pointing": C.pointing_hit(full, gt_mask), "energy": C.energy_fraction(full, gt_mask)})
        cams[j] = native
    items = pd.DataFrame(rows)

    rnd = copy.deepcopy(det)
    C.randomize(rnd.backbone.body.layer2)
    sanity = []
    for j in sanity_indices(len(boxes)):
        x, _, _ = load(boxes[j]["image_id"])
        _, native_r = C.cam_detector_box(rnd, x, torch.from_numpy(boxes[j]["box"]).to(dev), return_native=True)
        sanity.append({"image_id": boxes[j]["image_id"], "spearman": C.spearman(cams[j], native_r)})

    # Contact sheet: 10 true positives and 10 false positives at conf 0.25, crops around the box.
    kept = preds[preds.confidence >= CV2_CONF]
    fps = kept[~kept.matched].sample(n=min(10, int((~kept.matched).sum())), random_state=SHEET_SEED)
    tps = pd.DataFrame([{"k": j} for j in range(len(boxes))]).sample(n=min(10, len(boxes)), random_state=SHEET_SEED)
    panels = []
    for kind, entries in (("TP", [(boxes[j]["image_id"], boxes[j]["box"]) for j in tps.k]),
                          ("FP", [(r.image_id, np.array([r.x1, r.y1, r.x2, r.y2], np.float32)) for r in fps.itertuples()])):
        for image_id, box in entries:
            x, rgb, gt = load(image_id)
            full, _ = C.cam_detector_box(det, x, torch.from_numpy(box).to(dev), return_native=True)
            img = C.overlay(rgb, full)
            for g in gt:
                img = C.draw_box(img, g, (0, 255, 0))
            img = C.draw_box(img, box, (255, 0, 0))
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            half = max(64, 4 * max(box[2] - box[0], box[3] - box[1]))
            H, W = rgb.shape[:2]
            x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
            x2, y2 = int(min(W, cx + half)), int(min(H, cy + half))
            panels.append((img[y1:y2, x1:x2], f"{kind} {image_id}"))
    contact_sheet(panels, out / "attention_cv2.jpg")

    finish(out, "cv2", items, pd.DataFrame(sanity),
           {"eval_set": f"iToBoS val, true-positive boxes at conf {CV2_CONF}",
            "method": "Grad-CAM per box, lesion logit through RoI-Align, backbone layer2 (stride 8)",
            "energy_denominator": "whole image"})


def main() -> None:
    args = parse_args()
    out = args.run_root / "attention" / args.task
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    {"cv2": run_cv2, "cv3": run_cv3, "cv4": run_cv4}[args.task](args, out)


if __name__ == "__main__":
    main()
