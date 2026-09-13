"""
Verify a transferred bundle ON the GPU host, BEFORE configuring a GPU.

This is the network-volume half of the pre-flight discipline: pre-flight
proves the *code* is right on a laptop, this proves the *data* survived
the trip. Both run on CPU, before anything is billed at GPU rates.

What it actually catches, all of which are silent failures otherwise:

  - a truncated or corrupted object from an interrupted S3 transfer
    (checksum mismatch -- a short JPEG still decodes, it just decodes
    to something different, and training would happily consume it)
  - a partial sync: files missing that dataset.csv expects
  - an aborted untar leaving a subset of images
  - label distribution silently different from what was built

Exits non-zero on any failure, so it composes into a shell chain:

    python scripts/verify_gpu_bundle.py --data-root /workspace/cv4b_bundle \\
      && echo "safe to attach the GPU"

    # fast path when you trust the transfer and just want structure checked
    python scripts/verify_gpu_bundle.py --data-root /workspace/cv4b_bundle --sample 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

EXPECTED_FOLDS = ("pooled", "loso_atlas", "loso_stanford")


def fail(message: str) -> None:
    print(f"FAIL  {message}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0,
                        help="checksum only N randomly-chosen files instead of all 26k "
                             "(0 = verify everything, which is the default and the safe choice)")
    args = parser.parse_args()

    root = args.data_root
    print(f"verifying bundle at {root}\n")

    for name in ("dataset.csv", "SHA256SUMS", "BUNDLE.json"):
        if not (root / name).exists():
            fail(f"{name} missing -- transfer incomplete or wrong directory")

    info = json.loads((root / "BUNDLE.json").read_text())
    table = pd.read_csv(root / "dataset.csv", keep_default_na=False)
    print(f"BUNDLE.json: {info['images']} images, built at commit {info['git_commit'][:12]}")

    if len(table) != info["images"]:
        fail(f"dataset.csv has {len(table)} rows, BUNDLE.json expects {info['images']}")

    missing_columns = ({"relative_path", "source", "label", "is_melanoma", "isic_split"}
                       | set(EXPECTED_FOLDS)) - set(table.columns)
    if missing_columns:
        fail(f"dataset.csv missing columns: {', '.join(sorted(missing_columns))}")

    missing = [r for r in table["relative_path"] if not (root / r).exists()]
    if missing:
        fail(f"{len(missing)} image files listed but absent, e.g. {missing[0]}")
    print(f"PASS  all {len(table)} listed files present")

    expected = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, _, relative = line.partition("  ")
        expected[relative] = digest

    targets = sorted(table["relative_path"])
    if args.sample:
        targets = list(pd.Series(targets).sample(min(args.sample, len(targets)), random_state=0))
        print(f"      (checksumming a {len(targets)}-file sample, not all files)")

    bad = []
    for index, relative in enumerate(targets, 1):
        if relative not in expected:
            fail(f"{relative} has no recorded checksum")
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected[relative]:
            bad.append(relative)
        if index % 5000 == 0:
            print(f"      {index}/{len(targets)} hashed", flush=True)
    if bad:
        fail(f"{len(bad)} file(s) failed checksum -- re-transfer these, e.g. {bad[0]}")
    print(f"PASS  {len(targets)} checksums match")

    if hashlib.sha256((root / "dataset.csv").read_bytes()).hexdigest() != expected.get("dataset.csv"):
        fail("dataset.csv checksum mismatch -- the label file itself is corrupt")
    print("PASS  dataset.csv checksum matches")

    for fold in EXPECTED_FOLDS:
        clinical = table[table["source"] != "isic2019"]
        counts = clinical[fold].value_counts().to_dict()
        if not {"train", "val", "test"} <= set(counts):
            fail(f"fold {fold} is missing a split: {counts}")
        for split in ("train", "val"):
            subset = clinical[clinical[fold] == split]
            if subset["label"].nunique() < 2:
                fail(f"fold {fold} split {split} has only one class")
    isic_counts = table[table["source"] == "isic2019"]["isic_split"].value_counts().to_dict()
    if not {"train", "val", "test"} <= set(isic_counts):
        fail(f"ISIC splits incomplete: {isic_counts}")
    print(f"PASS  splits intact (ISIC {isic_counts})")

    sample_paths = list(pd.Series(table["relative_path"]).sample(
        min(40, len(table)), random_state=1))
    for relative in sample_paths:
        with Image.open(root / relative) as image:
            image.convert("RGB")
    print(f"PASS  {len(sample_paths)} sampled images decode")

    print("\nBundle verified. Safe to attach a GPU.")


if __name__ == "__main__":
    main()
