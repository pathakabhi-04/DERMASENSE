"""
Build the self-contained bundle that gets uploaded to the GPU host.

## Why a bundle, rather than shipping the repo's data directories

The split CSVs and the pre-resize manifest both store **absolute local
paths** (`/home/abhinav-pathak/...`), and `isic_rows()` resolves ISIC
images through `CVDataset`, which builds paths against a raw-data root
that will not exist on a rented pod. Shipping those as-is means either a
crash on the first batch or -- worse -- needing the 19 GB raw ISIC tree
present just to construct filenames.

So the bundle is flat, relative, and self-describing:

    bundle/
      dataset.csv     one row per image: relative path, label, splits
      images/<source>/<stem>.jpg
      SHA256SUMS      integrity, verified after transfer
      BUNDLE.json     counts + provenance (git commit, source checkpoint)

`dataset.csv` carries every label and every split assignment, so nothing
on the GPU host needs `CVDataset`, the raw datasets, the pre-resize
manifest, or this machine's directory layout. Only the repo (from git)
and the bundle (from the network volume).

**The bundle is canonical for local runs too.** Pre-flight, training and
evaluation all read it, so what passes on this laptop is exactly what
runs on the pod -- rather than a second code path that is only exercised
for the first time on a paid GPU.

Images are hardlinked from the pre-resized directory where possible, so
this costs no extra disk and no re-encoding.

    PYTHONPATH=. python3 scripts/build_gpu_bundle.py
    PYTHONPATH=. python3 scripts/build_gpu_bundle.py --tar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

from scripts.finetune_cv4b_backbone import SOURCE_CHECKPOINT, SPLITS, isic_rows
from scripts.preresize_cv4b_dataset import MANIFEST as PRERESIZE_MANIFEST

DG_SPLITS = Path(__file__).resolve().parents[1] / "analysis/quality/mel_sensitivity/cv4b_dg_splits.csv"

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "data/processed/cv4b_bundle"
OLD_FOLDS = ("pooled", "loso_atlas", "loso_stanford")
DG_FOLDS = ("dg_atlas", "dg_stanford", "dg_pad")
FOLDS = OLD_FOLDS + DG_FOLDS


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except subprocess.SubprocessError:
        return "unknown"


def build_rows(resized_by_original: dict[str, str]) -> pd.DataFrame:
    """One row per image, carrying EVERY fold assignment.

    Two generations of folds coexist deliberately:
      pooled / loso_*  the first fine-tune (cv4b_finetune_splits.csv),
                       kept so that experiment stays reproducible
      dg_*             the multi-domain run (cv4b_dg_splits.csv), which
                       is authoritative for every source including ISIC
                       and PAD-UFES
    """
    dg = pd.read_csv(DG_SPLITS).set_index("image_path")
    clinical_folds = pd.read_csv(SPLITS).set_index("image_path")
    records = []

    for split in ("train", "val", "test"):
        for row in isic_rows(split):
            resized = resized_by_original.get(row.image_path)
            if resized is None:
                raise SystemExit(f"no pre-resized copy for ISIC image {row.image_path}")
            records.append({
                "relative_path": f"images/isic2019/{Path(resized).name}",
                "source_file": resized,
                "source": "isic2019",
                "domain_group": "isic",
                "label": row.label,
                "is_melanoma": row.is_melanoma,
                "isic_split": split,
                **{fold: "" for fold in OLD_FOLDS},
                # ISIC keeps its frozen split in every dg fold.
                **{fold: split for fold in DG_FOLDS},
            })

    for image_path, row in dg.iterrows():
        resized = resized_by_original.get(image_path)
        if resized is None:
            raise SystemExit(f"no pre-resized copy for {image_path}")
        old = clinical_folds.loc[image_path] if image_path in clinical_folds.index else None
        records.append({
            "relative_path": f"images/{row['source']}/{Path(resized).name}",
            "source_file": resized,
            "source": row["source"],
            "domain_group": row["domain_group"],
            "label": int(bool(row["is_malignant"])),
            "is_melanoma": bool(row["is_melanoma"]),
            "isic_split": "",
            **{fold: (old[fold] if old is not None else "") for fold in OLD_FOLDS},
            **{fold: row[fold] for fold in DG_FOLDS},
        })

    frame = pd.DataFrame(records)
    duplicated = frame["relative_path"].duplicated()
    if duplicated.any():
        raise SystemExit(
            f"{duplicated.sum()} duplicate bundle paths, e.g. "
            f"{frame.loc[duplicated, 'relative_path'].iloc[0]}"
        )
    return frame


def place_images(frame: pd.DataFrame) -> None:
    for record in frame.itertuples():
        destination = BUNDLE_ROOT / record.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            continue
        try:
            os.link(record.source_file, destination)
        except OSError:
            shutil.copy2(record.source_file, destination)


def write_checksums(frame: pd.DataFrame) -> Path:
    """SHA256 over every image plus dataset.csv. An S3 round trip that
    silently truncates a file is otherwise invisible until training
    produces quietly wrong numbers."""
    lines = []
    for index, relative in enumerate(sorted(frame["relative_path"]), 1):
        digest = hashlib.sha256((BUNDLE_ROOT / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}")
        if index % 5000 == 0:
            print(f"  hashed {index}/{len(frame)}", flush=True)
    for extra in ("dataset.csv", f"checkpoint/{SOURCE_CHECKPOINT.name}"):
        digest = hashlib.sha256((BUNDLE_ROOT / extra).read_bytes()).hexdigest()
        lines.append(f"{digest}  {extra}")

    path = BUNDLE_ROOT / "SHA256SUMS"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", action="store_true",
                        help="also write cv4b_bundle.tar next to the bundle "
                             "(one big object transfers far faster than 26k small ones)")
    args = parser.parse_args()

    if not PRERESIZE_MANIFEST.exists():
        raise SystemExit(
            f"{PRERESIZE_MANIFEST} missing; run scripts/preresize_cv4b_dataset.py first"
        )
    manifest = pd.read_csv(PRERESIZE_MANIFEST)
    resized_by_original = dict(zip(manifest["image_path"], manifest["resized_path"]))

    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    frame = build_rows(resized_by_original)
    print(f"{len(frame)} images ({(frame['source'] == 'isic2019').sum()} ISIC, "
          f"{(frame['source'] != 'isic2019').sum()} clinical)")

    # The fine-tune starts FROM the shipped checkpoint, and `checkpoints/`
    # is gitignored -- so cloning the repo on a pod does not bring it. It
    # ships in the bundle or the run dies on the first line of
    # build_model(), after the data is uploaded and the GPU is rented.
    if not SOURCE_CHECKPOINT.exists():
        raise SystemExit(f"source checkpoint missing: {SOURCE_CHECKPOINT}")
    checkpoint_destination = BUNDLE_ROOT / "checkpoint" / SOURCE_CHECKPOINT.name
    checkpoint_destination.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_destination.exists():
        shutil.copy2(SOURCE_CHECKPOINT, checkpoint_destination)
    print(f"source checkpoint bundled: {checkpoint_destination.name} "
          f"({checkpoint_destination.stat().st_size / 1024 / 1024:.0f} MB)")

    place_images(frame)
    frame.drop(columns=["source_file"]).to_csv(BUNDLE_ROOT / "dataset.csv", index=False)

    print("hashing for transfer integrity...")
    write_checksums(frame)

    total_bytes = sum(
        (BUNDLE_ROOT / relative).stat().st_size for relative in frame["relative_path"]
    )
    (BUNDLE_ROOT / "BUNDLE.json").write_text(json.dumps({
        "images": len(frame),
        "bytes": total_bytes,
        "by_source": frame["source"].value_counts().to_dict(),
        "folds": list(FOLDS),
        "git_commit": git_commit(),
        "source_checkpoint": f"checkpoint/{SOURCE_CHECKPOINT.name}",
        "note": "self-contained; relative paths only. See scripts/build_gpu_bundle.py",
    }, indent=2))

    print(f"\nbundle -> {BUNDLE_ROOT}  ({total_bytes / 1024 / 1024:.0f} MB)")

    if args.tar:
        tar_path = BUNDLE_ROOT.parent / "cv4b_bundle.tar"
        print(f"writing {tar_path} ...")
        with tarfile.open(tar_path, "w") as archive:
            archive.add(BUNDLE_ROOT, arcname="cv4b_bundle")
        print(f"tar -> {tar_path}  ({tar_path.stat().st_size / 1024 / 1024:.0f} MB)")


if __name__ == "__main__":
    main()
