"""Build the flat, relative data bundle for the joint seg+cls experiment.

Local only (reads data/raw). Output: data/processed/academic_joint_bundle/

    ham/images/{id}.jpg        HAM10000 images, byte-for-byte copies (600x450)
    ham/masks/{id}.png         Tschandl masks, re-encoded as L-mode {0,255}
    isic2018/images/{id}.png   ISIC 2018 Task 1 test, resized to 512x512
    isic2018/masks/{id}.png      (cv2 INTER_LINEAR / INTER_NEAREST, exactly
                                 the resize CV-3 evaluation applies)
    pad/images/{id}.png        PAD-UFES train/val/test, resized to 224x224
                                 (PIL bilinear, the C1 eval resize)
    dataset.csv, SHA256SUMS, BUNDLE.json

HAM membership and split come from data/splits/isic2019 (seed 42) restricted
to images that have a Tschandl mask. The run aborts on any count mismatch
with spec Section 3.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
MASK_DIR = RAW / "ham10000_masks/HAM10000_segmentations_lesion_tschandl"
OUT = ROOT / "data/processed/academic_joint_bundle"

EXPECTED_HAM = {
    "train": {"AK": 92, "BCC": 377, "BKL": 777, "DF": 78, "MEL": 766, "NV": 4695, "SCC": 146, "VASC": 100},
    "val": {"AK": 32, "BCC": 70, "BKL": 160, "DF": 20, "MEL": 195, "NV": 1007, "SCC": 25, "VASC": 24},
    "test": {"AK": 6, "BCC": 67, "BKL": 162, "DF": 17, "MEL": 152, "NV": 1003, "SCC": 26, "VASC": 18},
}
EXPECTED_ISIC2018_TEST = 260
EXPECTED_PAD = {"train": 1610, "val": 336, "test": 352}


def binary_mask(path: Path) -> np.ndarray:
    # convert("L") first: 402 Tschandl PNGs have an inverted palette.
    with Image.open(path) as im:
        return (np.array(im.convert("L")) > 127).astype(np.uint8) * 255


def build_ham(rows: list[dict]) -> None:
    ham_ids = {p.name.removesuffix("_segmentation.png") for p in MASK_DIR.glob("*_segmentation.png")}
    (OUT / "ham/images").mkdir(parents=True)
    (OUT / "ham/masks").mkdir(parents=True)
    for split in ("train", "val", "test"):
        df = pd.read_csv(ROOT / f"data/splits/isic2019/{split}.csv")
        df = df[df["image"].isin(ham_ids)]
        got = df["native_diagnosis"].value_counts().to_dict()
        if got != EXPECTED_HAM[split]:
            raise SystemExit(f"HAM {split} counts {got} != spec {EXPECTED_HAM[split]}")
        for r in df.itertuples():
            src = RAW / "isic2019" / r.archive_path
            shutil.copyfile(src, OUT / f"ham/images/{r.image}.jpg")
            m = binary_mask(MASK_DIR / f"{r.image}_segmentation.png")
            with Image.open(src) as im:
                if im.size != (m.shape[1], m.shape[0]):
                    raise SystemExit(f"image/mask size mismatch for {r.image}")
            Image.fromarray(m, mode="L").save(OUT / f"ham/masks/{r.image}.png")
            rows.append({"source": "ham", "split": split, "image_id": r.image, "label": r.native_diagnosis,
                         "lesion_id": r.lesion_id, "image": f"ham/images/{r.image}.jpg",
                         "mask": f"ham/masks/{r.image}.png"})
        print(f"ham {split}: {len(df)}")


def build_isic2018(rows: list[dict]) -> None:
    (OUT / "isic2018/images").mkdir(parents=True)
    (OUT / "isic2018/masks").mkdir(parents=True)
    df = pd.read_csv(ROOT / "data/splits/isic2018_task1/test.csv")
    if len(df) != EXPECTED_ISIC2018_TEST:
        raise SystemExit(f"ISIC 2018 test has {len(df)} rows, expected {EXPECTED_ISIC2018_TEST}")
    for r in df.itertuples():
        img = cv2.imread(str(ROOT / r.image_path), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(ROOT / r.mask_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT / f"isic2018/images/{r.image_id}.png"), img)
        cv2.imwrite(str(OUT / f"isic2018/masks/{r.image_id}.png"), ((msk > 127) * 255).astype(np.uint8))
        rows.append({"source": "isic2018", "split": "test", "image_id": r.image_id, "label": "",
                     "lesion_id": "", "image": f"isic2018/images/{r.image_id}.png",
                     "mask": f"isic2018/masks/{r.image_id}.png"})
    print(f"isic2018 test: {len(df)}")


def build_pad(rows: list[dict]) -> None:
    (OUT / "pad/images").mkdir(parents=True)
    for split, expected in EXPECTED_PAD.items():
        df = pd.read_csv(ROOT / f"data/splits/pad_ufes/{split}.csv")
        if len(df) != expected:
            raise SystemExit(f"PAD {split} has {len(df)} rows, expected {expected}")
        for r in df.itertuples():
            stem = Path(r.image_id).stem
            with Image.open(ROOT / r.image_path) as im:
                im.convert("RGB").resize((224, 224), Image.BILINEAR).save(OUT / f"pad/images/{stem}.png")
            rows.append({"source": "pad", "split": split, "image_id": r.image_id, "label": r.native_diagnosis,
                         "lesion_id": r.lesion_uid, "image": f"pad/images/{stem}.png", "mask": ""})
        print(f"pad {split}: {len(df)}")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; delete it to rebuild.")
    OUT.mkdir(parents=True)
    rows: list[dict] = []
    build_ham(rows)
    build_isic2018(rows)
    build_pad(rows)

    df = pd.DataFrame(rows)
    if df["image"].duplicated().any():
        raise SystemExit("duplicate bundle paths")
    df.to_csv(OUT / "dataset.csv", index=False)

    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS", "BUNDLE.json"))
    with open(OUT / "SHA256SUMS", "w") as fh:
        for p in files:
            fh.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}\n")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    summary = {
        "name": "academic_joint_bundle",
        "spec": "docs/academic_joint_seg_cls_spec.md",
        "built_from_commit": commit,
        "files": len(files) + 1,  # + SHA256SUMS
        "bytes": sum(p.stat().st_size for p in files),
        "counts": df.groupby(["source", "split"]).size().rename("n").reset_index().to_dict("records"),
    }
    (OUT / "BUNDLE.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
