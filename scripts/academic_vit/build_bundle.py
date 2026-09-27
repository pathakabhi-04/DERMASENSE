"""ISIC 2019 bundle for docs/academic_vit_spec.md (§3.1).

Reuses main's validated cv4b pre-resize (320x320, LANCZOS, JPEG q100) and adds
the classes that bundle lacks (DF and VASC, which have no PAD-UFES analogue),
so all eight classes share one preprocessing. Local only.

    data/processed/academic_vit_bundle/
      isic2019/{image_id}.jpg     hardlinked from cv4b_bundle, or resized here
      dataset.csv, SHA256SUMS, BUNDLE.json
    data/processed/academic_vit_bundle.tar   (upload this single object)

resize_one() is a verbatim copy of scripts/preresize_cv4b_dataset.py on main
(importing it would pull in main's training stack). Before anything is
written, it must reproduce 50 existing cv4b images byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/isic2019"
CV4B = Path.home() / "dermasense/data/processed/cv4b_bundle"
OUT = ROOT / "data/processed/academic_vit_bundle"
EXPECTED = {"train": 18402, "val": 3375, "test": 3554}

# --- verbatim from main: scripts/preresize_cv4b_dataset.py -----------------
TARGET_SIZE = (320, 320)
JPEG_QUALITY = 100
RESAMPLE = Image.LANCZOS


def resize_one(task: tuple[str, str]) -> tuple[str, str, bool]:
    source, destination = task
    destination_path = Path(destination)
    if destination_path.exists():
        return source, destination, True
    try:
        with Image.open(source) as image:
            resized = image.convert("RGB").resize(TARGET_SIZE, RESAMPLE)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(destination_path, "JPEG", quality=JPEG_QUALITY)
        return source, destination, True
    except Exception:  # noqa: BLE001 -- one bad file must not kill a 27k-image job
        return source, destination, False
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; delete it to rebuild.")
    rows = []
    for split, n in EXPECTED.items():
        df = pd.read_csv(ROOT / f"data/splits/isic2019/{split}.csv")
        if len(df) != n:
            raise SystemExit(f"{split}: {len(df)} rows, expected {n}")
        for r in df.itertuples():
            rows.append({"source": "isic2019", "split": split, "image_id": r.image, "label": r.native_diagnosis,
                         "lesion_id": r.lesion_id, "image": f"isic2019/{r.image}.jpg",
                         "original": str(RAW / r.archive_path),
                         "cv4b": CV4B / "images/isic2019" / f"{Path(r.archive_path).stem}.jpg"})
    frame = pd.DataFrame(rows)
    in_cv4b = frame["cv4b"].map(Path.exists)
    print(f"{int(in_cv4b.sum())} images already in cv4b_bundle, {int((~in_cv4b).sum())} to resize "
          f"({frame.loc[~in_cv4b, 'label'].value_counts().to_dict()})")

    # Gate: the copied function must reproduce main's files exactly.
    check_dir = OUT.parent / "_resize_check"
    check_dir.mkdir(parents=True, exist_ok=True)
    sample = frame[in_cv4b].sample(n=50, random_state=42)
    for r in sample.itertuples():
        dest = check_dir / f"{r.image_id}.jpg"
        dest.unlink(missing_ok=True)
        resize_one((r.original, str(dest)))
        if dest.read_bytes() != Path(r.cv4b).read_bytes():
            raise SystemExit(f"resize_one does not reproduce cv4b bytes for {r.image_id}; aborting")
        dest.unlink()
    check_dir.rmdir()
    print("resize_one reproduces 50/50 cv4b images byte for byte")

    (OUT / "isic2019").mkdir(parents=True)
    for r in frame[in_cv4b].itertuples():
        os.link(r.cv4b, OUT / r.image)
    tasks = [(r.original, str(OUT / r.image)) for r in frame[~in_cv4b].itertuples()]
    with ProcessPoolExecutor(max_workers=8) as pool:
        failed = [src for src, _, ok in pool.map(resize_one, tasks, chunksize=16) if not ok]
    if failed:
        raise SystemExit(f"{len(failed)} images failed to resize, e.g. {failed[0]}")

    frame["from_cv4b"] = in_cv4b
    frame.drop(columns=["original", "cv4b"]).to_csv(OUT / "dataset.csv", index=False)
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    with open(OUT / "SHA256SUMS", "w") as fh:
        for p in files:
            fh.write(f"{sha256(p)}  {p.relative_to(OUT)}\n")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    summary = {"name": "academic_vit_bundle", "spec": "docs/academic_vit_spec.md", "built_from_commit": commit,
               "preprocessing": "320x320 LANCZOS JPEG q100 (main: scripts/preresize_cv4b_dataset.py)",
               "images": len(frame), "from_cv4b": int(in_cv4b.sum()), "resized_here": int((~in_cv4b).sum()),
               "files": len(files) + 2, "bytes": sum(p.stat().st_size for p in files),
               "counts": frame.groupby(["split", "label"]).size().rename("n").reset_index().to_dict("records")}
    (OUT / "BUNDLE.json").write_text(json.dumps(summary, indent=2))

    with tarfile.open(str(OUT) + ".tar", "w") as tar:
        tar.add(OUT, arcname=OUT.name)
    print(json.dumps({k: v for k, v in summary.items() if k != "counts"}, indent=2))


if __name__ == "__main__":
    main()
