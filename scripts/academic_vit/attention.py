"""Experiment 3 attention maps: rollout (primary) and Grad-CAM (bridge) for
the ViT CV-4 and CV-3 models, scored exactly as Experiment 2
(docs/academic_vit_spec.md §6). Scoring, sanity check and figures reuse
scripts/academic_r50/attention.py. Writes <run-root>/attention/<task>_<map>/.

    PYTHONPATH=. python3 scripts/academic_vit/attention.py --task cv4 \
        --ham-root /workspace/dermasense_academic/data/academic_joint_bundle \
        --r50-root /workspace/dermasense_academic/data/academic_r50_bundle \
        --run-root /workspace/dermasense_academic/runs/academic_vit
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.academic import cam as C
from src.academic.data import IMAGENET_MEAN, IMAGENET_STD, ISIC2019_CLASSES, SegTestDataset, load_mask, read_bundle
from src.academic.model import AcademicModel
from src.academic.vit import ViTClassifier

_spec = importlib.util.spec_from_file_location(
    "r50_attention", Path(__file__).resolve().parents[1] / "academic_r50/attention.py")
R50 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R50)

SHEET_IDS = Path(__file__).resolve().parents[2] / "evaluation/academic_r50/sheet_ids.json"
MAPS = ("rollout", "gradcam")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=("cv3", "cv4"), required=True)
    p.add_argument("--ham-root", type=Path, help="academic_joint_bundle (cv4)")
    p.add_argument("--r50-root", type=Path, help="academic_r50_bundle (cv3)")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="Smoke tests only.")
    return p.parse_args()


def score(up: np.ndarray, gt: np.ndarray) -> dict:
    return {"pointing": C.pointing_hit(up, gt), "energy": C.energy_fraction(up, gt), "iou": C.iou_at_half(up, gt)}


def run_cv4(a, dev) -> None:
    ck = torch.load(a.run_root / "cv4/best.pt", map_location="cpu", weights_only=False)
    model = ViTClassifier(pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    df = read_bundle(a.ham_root)
    df = df[(df.source == "ham") & (df.split == "test")].reset_index(drop=True)
    if a.limit:
        df = df.head(a.limit)

    def load(i):
        row = df.iloc[i]
        with Image.open(a.ham_root / row["image"]) as im:
            im = im.convert("RGB")
            x, rgb = tf(im)[None].to(dev), np.array(im.resize((224, 224), Image.BILINEAR))
        gt = np.array(load_mask(a.ham_root / row["mask"]).resize((224, 224), Image.NEAREST)) > 127
        return x, rgb, gt

    rows = {m: [] for m in MAPS}
    cams, preds = {m: {} for m in MAPS}, {}
    for i in range(len(df)):
        x, _, gt = load(i)
        g, pred = C.cam_vit_classifier(model, x)
        r = C.rollout_classifier(model, x)
        preds[i] = pred
        base = {"image_id": df.image_id[i], "true": df.label[i], "pred": ISIC2019_CLASSES[pred],
                "correct": ISIC2019_CLASSES.index(df.label[i]) == pred}
        for m, cam in (("rollout", r), ("gradcam", g)):
            rows[m].append({**base, **score(C.upsample(cam, (224, 224)), gt)})
            cams[m][i] = cam

    rnd = copy.deepcopy(model)
    C.randomize(rnd.vit.encoder.layers[-1])
    sanity = {m: [] for m in MAPS}
    for i in R50.sanity_indices(len(df)):
        x, _, _ = load(i)
        g_r, _ = C.cam_vit_classifier(rnd, x, target=preds[i])
        r_r = C.rollout_classifier(rnd, x)
        sanity["gradcam"].append({"image_id": df.image_id[i], "spearman": C.spearman(cams["gradcam"][i], g_r)})
        sanity["rollout"].append({"image_id": df.image_id[i], "spearman": C.spearman(cams["rollout"][i], r_r)})

    ids = json.loads(SHEET_IDS.read_text())["cv4"]
    index = {v: k for k, v in df.image_id.items()}
    for m in MAPS:
        out = a.run_root / "attention" / f"cv4_{m}"
        out.mkdir(parents=True, exist_ok=True)
        panels = []
        for image_id in ids:
            if image_id not in index:
                continue
            i = index[image_id]
            _, rgb, gt = load(i)
            it = rows[m][i]
            img = C.draw_outline(C.overlay(rgb, C.upsample(cams[m][i], (224, 224))), gt)
            panels.append((img, f"{it['true']}->{it['pred']} {'ok' if it['correct'] else 'WRONG'}"))
        R50.contact_sheet(panels, out / f"attention_cv4_{m}.jpg")
        items = pd.DataFrame(rows[m])
        R50.finish(out, f"cv4_{m}", items, pd.DataFrame(sanity[m]),
                   {"eval_set": "HAM10000 in ISIC 2019 test (originals, as Experiment 2)",
                    "method": {"rollout": "attention rollout, CLS row (14x14 at 224)",
                               "gradcam": "Grad-CAM, predicted-class logit, last block tokens (14x14)"}[m],
                    "sheet_images": "Experiment 2 CV-4 sheet IDs (evaluation/academic_r50/sheet_ids.json)"})


def run_cv3(a, dev) -> None:
    ck = torch.load(a.run_root / "S_seed42/best.pt", map_location="cpu", weights_only=False)
    model = AcademicModel("S", pretrained=False, seed=42, backbone="vit_b_16", image_size=512)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    ds = SegTestDataset(a.r50_root, image_size=512, split="test")
    n = min(len(ds), a.limit or len(ds))

    def load(i):
        item = ds[i]
        with Image.open(a.r50_root / ds.df.iloc[i]["image"]) as im:
            rgb = np.array(im.convert("RGB"))
        return item["image"][None].to(dev), rgb, item["mask"][0].numpy() > 0.5

    rows = {m: [] for m in MAPS}
    cams, regions, empty = {m: {} for m in MAPS}, {}, []
    for i in range(n):
        x, _, gt = load(i)
        g, region = C.cam_vit_segmenter(model, x)
        r = None if g is None else C.rollout_segmenter(model, x, region)
        if g is None or r is None:
            empty.append(ds.df.image_id[i])
            continue
        regions[i] = region
        for m, cam in (("rollout", r), ("gradcam", g)):
            rows[m].append({"image_id": ds.df.image_id[i], **score(C.upsample(cam, (512, 512)), gt)})
            cams[m][i] = cam

    rnd = copy.deepcopy(model)
    C.randomize(rnd.encoder.vit.encoder.layers[-1])
    valid = sorted(regions)
    sanity = {m: [] for m in MAPS}
    for k in R50.sanity_indices(len(valid)):
        i = valid[k]
        x, _, _ = load(i)
        g_r, _ = C.cam_vit_segmenter(rnd, x, region=regions[i])
        r_r = C.rollout_segmenter(rnd, x, regions[i])
        sanity["gradcam"].append({"image_id": ds.df.image_id[i], "spearman": 0.0 if g_r is None else C.spearman(cams["gradcam"][i], g_r)})
        sanity["rollout"].append({"image_id": ds.df.image_id[i], "spearman": 0.0 if r_r is None else C.spearman(cams["rollout"][i], r_r)})

    ids = json.loads(SHEET_IDS.read_text())["cv3"]
    index = {v: k for k, v in ds.df.image_id.items()}
    for m in MAPS:
        out = a.run_root / "attention" / f"cv3_{m}"
        out.mkdir(parents=True, exist_ok=True)
        panels = []
        for image_id in ids:
            i = index.get(image_id)
            if i is None or i not in regions:
                continue
            _, rgb, gt = load(i)
            img = C.draw_outline(C.overlay(rgb, C.upsample(cams[m][i], (512, 512))), gt)
            img = C.draw_outline(img, regions[i][0, 0].cpu().numpy(), color=(255, 255, 255))
            panels.append((img, image_id))
        R50.contact_sheet(panels, out / f"attention_cv3_{m}.jpg")
        R50.finish(out, f"cv3_{m}", pd.DataFrame(rows[m]), pd.DataFrame(sanity[m]),
                   {"eval_set": "ISIC 2018 Task 1 test", "empty_prediction_excluded": len(empty),
                    "empty_prediction_ids": empty,
                    "method": {"rollout": "attention rollout, mean row of patches inside predicted mask (32x32 at 512)",
                               "gradcam": "Seg-Grad-CAM, sum of mask logits in predicted mask, last block tokens (32x32)"}[m]})


def main() -> None:
    a = parse_args()
    torch.manual_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (run_cv4 if a.task == "cv4" else run_cv3)(a, dev)


if __name__ == "__main__":
    main()
