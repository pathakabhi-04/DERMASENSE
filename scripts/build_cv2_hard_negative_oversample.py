"""
CV-2 Experiment D: build the sun-damage hard-negative oversampled training list.

Purpose (single, bounded change from B1 -- see project decision record):
The pathological zero-lesion image audit (analysis/quality/
cv2_pathological_audit/) found that images with sun_damage_level >= 2 are
present in the full zero-lesion training population at ~5.8% (94 + 3 out
of 1750), while the pathological false-positive-heavy images identified
by manual visual inspection show sun_damage_level >= 2 at 30% -- a ~5x
elevation. This script oversamples that ~5.8% subgroup by a fixed factor
in the training image list, with everything else (architecture,
resolution, augmentation, hyperparameters) left identical to B1.

This does NOT modify any training code. Ultralytics YOLO builds its
dataset from the plain-text image path list; duplicating a path in that
list increases how often that image (and its label file) is sampled per
epoch. This keeps Experiment D a single, isolated, auditable variable
change from B1, per the CV-2 spec's product-oriented priority (Section 21)
and the project's "one variable at a time" experimental discipline.

Usage (run locally, no GPU required):
    python scripts/build_cv2_hard_negative_oversample.py [--factor 8]

Output:
    data/splits/itobos_detection/hard_negative_ids.csv
        Versioned, auditable list of the exact image IDs identified as
        the oversampling target. Committed to git so the exact set used
        in Experiment D is reproducible and reviewable later.

    data/splits/itobos_detection/train_oversampled_sundamage.txt
        A copy of train.txt with each hard-negative image's path
        duplicated (factor - 1) additional times.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = REPO_ROOT / "data/raw/itobos/_train/_train/metadata.csv"
LABEL_DIR = REPO_ROOT / "data/raw/itobos/_train/_train/labels"
TRAIN_LIST_PATH = REPO_ROOT / "data/splits/itobos_detection/train.txt"

HARD_NEGATIVE_IDS_PATH = (
    REPO_ROOT / "data/splits/itobos_detection/hard_negative_ids.csv"
)
OVERSAMPLED_TRAIN_LIST_PATH = (
    REPO_ROOT
    / "data/splits/itobos_detection/train_oversampled_sundamage.txt"
)

SUN_DAMAGE_THRESHOLD = 2  # >= this level is the oversampling target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the sun-damage hard-negative oversampled training "
            "list for CV-2 Experiment D."
        )
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=8,
        help=(
            "Total effective representation multiplier for hard-negative "
            "images (default: 8x, i.e. 7 additional duplicate entries "
            "per image)."
        ),
    )
    return parser.parse_args()


def find_zero_lesion_ids(label_dir: Path) -> set[str]:
    zero_ids = set()
    for f in os.listdir(label_dir):
        if not f.endswith(".txt"):
            continue
        full_path = label_dir / f
        if full_path.stat().st_size == 0:
            zero_ids.add(f[: -len(".txt")])
    return zero_ids


def main() -> None:
    args = parse_args()

    if args.factor < 1:
        raise ValueError("--factor must be >= 1")

    for path in (METADATA_PATH, LABEL_DIR, TRAIN_LIST_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    meta = pd.read_csv(METADATA_PATH)
    zero_lesion_ids = find_zero_lesion_ids(LABEL_DIR)

    zero_meta = meta[meta["image_id"].isin(zero_lesion_ids)].copy()
    hard_negatives = zero_meta[
        zero_meta["sun_damage_level"] >= SUN_DAMAGE_THRESHOLD
    ].copy()

    hard_negative_ids = set(hard_negatives["image_id"])

    print("=" * 70)
    print("CV-2 EXPERIMENT D: HARD-NEGATIVE OVERSAMPLE BUILD")
    print("=" * 70)
    print(f"Total zero-lesion images in train: {len(zero_lesion_ids)}")
    print(
        f"Hard negatives (sun_damage_level >= {SUN_DAMAGE_THRESHOLD}): "
        f"{len(hard_negative_ids)}"
    )
    print(f"Oversample factor: {args.factor}x")
    print()

    hard_negatives[
        ["image_id", "age_at_baseline", "body_part", "sun_damage_level"]
    ].sort_values("image_id").to_csv(HARD_NEGATIVE_IDS_PATH, index=False)
    print(f"Hard-negative ID list saved to: {HARD_NEGATIVE_IDS_PATH}")

    with TRAIN_LIST_PATH.open("r", encoding="utf-8") as f:
        train_lines = [line.rstrip("\n") for line in f if line.strip()]

    original_count = len(train_lines)

    matched_lines = 0
    output_lines: list[str] = []

    for line in train_lines:
        output_lines.append(line)

        stem = Path(line).stem
        if stem in hard_negative_ids:
            matched_lines += 1
            # Add (factor - 1) additional copies for a total of `factor`.
            for _ in range(args.factor - 1):
                output_lines.append(line)

    if matched_lines != len(hard_negative_ids):
        print(
            "WARNING: matched "
            f"{matched_lines} lines in train.txt against "
            f"{len(hard_negative_ids)} hard-negative IDs. "
            "This mismatch should be understood before training -- "
            "it likely means some hard-negative images are not in the "
            "training split (e.g. they fell into val), which is fine, "
            "but confirm before proceeding."
        )

    with OVERSAMPLED_TRAIN_LIST_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")

    new_count = len(output_lines)
    effective_share = (
        matched_lines * args.factor / new_count if new_count else 0.0
    )

    print()
    print(f"Original train.txt lines:   {original_count}")
    print(f"Matched hard-negative lines: {matched_lines}")
    print(f"New oversampled list lines:  {new_count}")
    print(
        f"Effective per-epoch share of hard negatives: "
        f"{effective_share:.1%}"
    )
    print()
    print(f"Oversampled train list saved to: {OVERSAMPLED_TRAIN_LIST_PATH}")
    print()
    print(
        "Next step: create configs/cv2_itobos_d1_oversample.yaml pointing "
        "'train:' at this new file, with everything else identical to "
        "configs/cv2_itobos.yaml."
    )


if __name__ == "__main__":
    main()