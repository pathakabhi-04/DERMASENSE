"""
Pre-resize every image the CV-4b fine-tune touches to 256x256, once, on
CPU (design §11).

Two measured reasons, not one:

1. **Upload.** ISIC2019 raw is 19 GB on an external HDD. Every image is
   squashed to 224x224 before it reaches the model anyway.
2. **Epoch time, which turned out to matter more.** Pre-flight check 9
   measured the loader at ~73 img/s with 4 workers, and a page-cache
   re-read at the *same* rate -- so the bottleneck is JPEG decode, not
   disk. Decode cost scales with pixel count, and these originals are
   ~400 KB / ~1 MP. Shrinking them is the direct lever on epoch time.

Geometry matches inference exactly: `transforms.Resize((256, 256))` is
the same aspect-squashing operation as the deployed
`Resize((224, 224))`, just at a larger size, so augmentation still has
headroom and the final 224 resize is unchanged. It is NOT pixel-identical
to augmenting the original -- that is precisely what pre-flight check 7
tests before this is allowed to be used.

Writes a manifest mapping original -> resized so nothing depends on
filename mangling being collision-free.

    PYTHONPATH=. python3 scripts/preresize_cv4b_dataset.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.finetune_cv4b_backbone import SPLITS, isic_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "data/processed/cv4b_finetune_256"
MANIFEST = OUT_ROOT / "manifest.csv"

# Chosen by measurement, not preference (pre-flight check 7). Against
# full-resolution features through the deployed transform, on 240 val
# images, mean cosine similarity of the 2048-d features:
#
#   256 BILINEAR q95   0.9633   <- the obvious first choice, and too lossy
#   256 LANCZOS  q95   0.9757
#   256 LANCZOS  q100  0.9876
#   320 LANCZOS  q100  0.9932   <- chosen
#   256 LANCZOS  PNG   0.9955   <- best, but PNG is ~10x the bytes
#
# Two separate effects, both real: BILINEAR without antialiasing loses
# detail on a large downscale, and JPEG quantisation moves features more
# than the resize does. Interesting aside, not pursued here: compression
# sensitivity of this backbone is one of the untested hypotheses left open
# by the composition audit, and Fitzpatrick17k is far more compressed than
# DDI.
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


def collect_tasks() -> list[tuple[str, str, str]]:
    """Returns (original, resized, source_label)."""
    tasks = []

    for split in ("train", "val", "test"):
        for row in isic_rows(split):
            stem = Path(row.image_path).stem
            tasks.append((row.image_path, str(OUT_ROOT / "isic2019" / f"{stem}.jpg"), "isic2019"))

    clinical = pd.read_csv(SPLITS)
    for row in clinical.itertuples():
        stem = Path(row.image_path).stem
        tasks.append((row.image_path, str(OUT_ROOT / row.source / f"{stem}.jpg"), row.source))

    seen = {}
    for original, resized, source in tasks:
        if resized in seen and seen[resized] != original:
            raise RuntimeError(
                f"destination collision: {resized} claimed by both {seen[resized]} and {original}"
            )
        seen[resized] = original
    return tasks


def main() -> None:
    tasks = collect_tasks()
    print(f"{len(tasks)} images -> {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    records, failures = [], []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for index, (source, destination, ok) in enumerate(
            pool.map(resize_one, [(t[0], t[1]) for t in tasks], chunksize=32), 1
        ):
            (records if ok else failures).append((source, destination))
            if index % 2000 == 0 or index == len(tasks):
                print(f"  {index}/{len(tasks)}", flush=True)

    if failures:
        print(f"\n{len(failures)} images failed, e.g. {failures[0][0]}")

    by_destination = {destination: source for source, destination, _ in tasks}
    frame = pd.DataFrame(
        [{"image_path": source, "resized_path": destination,
          "source": Path(destination).parent.name}
         for source, destination in records if by_destination.get(destination)]
    )
    frame.to_csv(MANIFEST, index=False)

    total_mb = sum(Path(p).stat().st_size for p in frame["resized_path"]) / 1024 / 1024
    print(f"\n{len(frame)} resized, {total_mb:.0f} MB total -> {MANIFEST}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
