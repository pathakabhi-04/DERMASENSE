"""Build the data bundle for docs/academic_resnet50_three_task_spec.md.

Local only. Output: data/processed/academic_r50_bundle/

    itobos/images/{id}.png       iToBoS train+val (symlinks to the raw PNGs;
    itobos/labels/{id}.txt         aws s3 sync uploads the files they point to)
    isic2018/images/{id}.png     ISIC 2018 Task 1 train/val/test, resized to
    isic2018/masks/{id}.png        512x512 with the cv2 resize CV-3 applies
    cv4/isic2019_resnet50_weighted_best.pt   the CV-4 checkpoint (maps only)
    dataset.csv, SHA256SUMS, BUNDLE.json

HAM10000 test images for the CV-4 maps are read from academic_joint_bundle,
which is already on the volume.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/academic_r50_bundle"
ITOBOS = ROOT / "data/raw/itobos/_train/_train"
CV4_CKPT = Path.home() / "dermasense/checkpoints/isic2019_resnet50_weighted_best.pt"

EXPECTED_ITOBOS = {"train": (6778, 23498, 1401), "val": (1695, 5905, 349)}
EXPECTED_ISIC2018 = {"train": 2074, "val": 259, "test": 260}


def count_boxes(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; delete it to rebuild.")
    for d in ("itobos/images", "itobos/labels", "isic2018/images", "isic2018/masks", "cv4"):
        (OUT / d).mkdir(parents=True)
    rows: list[dict] = []

    for split, (n_img, n_box, n_zero) in EXPECTED_ITOBOS.items():
        df = pd.read_csv(ROOT / f"data/splits/itobos_detection/{split}.csv")
        boxes = 0
        zero = 0
        for r in df.itertuples():
            img, lab = ITOBOS / f"images/{r.image_id}.png", ITOBOS / f"labels/{r.image_id}.txt"
            n = count_boxes(lab)
            if n != r.lesion_count:
                raise SystemExit(f"{r.image_id}: label has {n} boxes, split says {r.lesion_count}")
            boxes += n
            zero += n == 0
            os.symlink(img.resolve(), OUT / f"itobos/images/{r.image_id}.png")
            os.symlink(lab.resolve(), OUT / f"itobos/labels/{r.image_id}.txt")
            rows.append({"source": "itobos", "split": split, "image_id": r.image_id,
                         "image": f"itobos/images/{r.image_id}.png", "label": f"itobos/labels/{r.image_id}.txt",
                         "mask": "", "lesion_count": n, "density_bucket": r.density_bucket})
        if (len(df), boxes, zero) != (n_img, n_box, n_zero):
            raise SystemExit(f"iToBoS {split}: {(len(df), boxes, zero)} != {(n_img, n_box, n_zero)}")
        print(f"itobos {split}: {len(df)} images, {boxes} boxes, {zero} zero-lesion")

    for split, expected in EXPECTED_ISIC2018.items():
        df = pd.read_csv(ROOT / f"data/splits/isic2018_task1/{split}.csv")
        if len(df) != expected:
            raise SystemExit(f"ISIC 2018 {split}: {len(df)} != {expected}")
        for r in df.itertuples():
            img = cv2.imread(str(ROOT / r.image_path), cv2.IMREAD_COLOR)
            msk = cv2.imread(str(ROOT / r.mask_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, (512, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(OUT / f"isic2018/images/{r.image_id}.png"), img)
            cv2.imwrite(str(OUT / f"isic2018/masks/{r.image_id}.png"), ((msk > 127) * 255).astype(np.uint8))
            rows.append({"source": "isic2018", "split": split, "image_id": r.image_id,
                         "image": f"isic2018/images/{r.image_id}.png", "label": "",
                         "mask": f"isic2018/masks/{r.image_id}.png", "lesion_count": "", "density_bucket": ""})
        print(f"isic2018 {split}: {len(df)}")

    # The original pickles main's src.training.TrainingConfig, which does not
    # exist on this branch or the pod. Store the weights only, plus the
    # original's sha256 and metadata. (Run with ~/dermasense on PYTHONPATH.)
    import torch

    ckpt = torch.load(CV4_CKPT, map_location="cpu", weights_only=False)
    torch.save({"model_state_dict": ckpt["model_state_dict"],
                "source": str(CV4_CKPT),
                "source_sha256": hashlib.sha256(CV4_CKPT.read_bytes()).hexdigest(),
                "epoch": ckpt["epoch"], "val_macro_f1": ckpt["val_macro_f1"]},
               OUT / "cv4" / CV4_CKPT.name)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dataset.csv", index=False)

    files = sorted(p for p in OUT.rglob("*") if (p.is_file() or p.is_symlink()) and p.name not in ("SHA256SUMS", "BUNDLE.json"))
    with open(OUT / "SHA256SUMS", "w") as fh:
        for p in files:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            fh.write(f"{h.hexdigest()}  {p.relative_to(OUT)}\n")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    summary = {
        "name": "academic_r50_bundle",
        "spec": "docs/academic_resnet50_three_task_spec.md",
        "built_from_commit": commit,
        "files": len(files) + 1,
        "bytes": sum(p.stat().st_size for p in files),
        "counts": df.groupby(["source", "split"]).size().rename("n").reset_index().to_dict("records"),
    }
    (OUT / "BUNDLE.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
