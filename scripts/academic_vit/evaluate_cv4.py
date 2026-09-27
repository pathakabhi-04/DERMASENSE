"""Test evaluation of the Experiment 3 CV-4 ViT (docs/academic_vit_spec.md §5).

Scores best.pt once on the ISIC 2019 test split (3,554 bundle images, the
same inputs R50b was scored on): macro-F1, per-class F1, accuracy, and
melanoma recall with a Wilson interval.

    PYTHONPATH=. python3 scripts/academic_vit/evaluate_cv4.py \
        --data-root /workspace/dermasense_academic/data/academic_vit_bundle \
        --run-dir   /workspace/dermasense_academic/runs/academic_vit/cv4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.academic.data import ISIC2019_CLASSES
from src.academic.metrics import macro_f1, per_class_f1
from src.academic.vit import ViTClassifier

_here = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--num-workers", type=int, default=8)
    a = p.parse_args()
    train_cv4, score = _load("train_cv4"), _load("score_r50_on_bundle")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = ViTClassifier(pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    ds = train_cv4.IsicBundle(a.data_root, "test")
    ys, ps = [], []
    for x, y in DataLoader(ds, batch_size=64, num_workers=a.num_workers):
        ps.append(model(x.to(dev)).argmax(1).cpu())
        ys.append(y)
    y, pr = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    pd.DataFrame({"image_id": ds.df.image_id, "true": [ISIC2019_CLASSES[i] for i in y],
                  "pred": [ISIC2019_CLASSES[i] for i in pr]}).to_csv(a.run_dir / "test_per_image.csv", index=False)
    mel = ISIC2019_CLASSES.index("MEL")
    k, n = int(((y == mel) & (pr == mel)).sum()), int((y == mel).sum())
    res = {"n": int(len(y)), "macro_f1": macro_f1(y, pr, 8), "accuracy": float((y == pr).mean()),
           "f1_per_class": dict(zip(ISIC2019_CLASSES, map(float, per_class_f1(y, pr, 8)))),
           "mel_recall": k / n, "mel_recall_wilson95": score.wilson(k, n), "mel_n": n, "best_epoch": ck["epoch"]}
    (a.run_dir / "test_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
