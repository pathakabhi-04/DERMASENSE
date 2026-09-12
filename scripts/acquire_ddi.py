"""
Acquire DDI + DDI-2 (Stanford AIMI, Redivis) to the external drive.

Why: both are biopsy-proven CLINICAL (non-dermoscopic) photos with
diverse skin tones -- the closest public stand-in for the phone-photo
deployment distribution, which is the largest unmeasured gap in the CV
pipeline (docs/product_scope_and_readiness.md gate 2). Every strong
number we have (the CV-4b referral head, 0.9024 melanoma routing) is
measured on ISIC dermoscopy; this is the test set that tells us whether
it survives the domain shift.

This only ACQUIRES: writes images + a label manifest per dataset. It
trains nothing and evaluates nothing -- see scripts/evaluate_referral_on_ddi.py
for the bounded Step 1 measurement this feeds.

## Auth

Needs a Redivis API token (both datasets are access-gated per account,
already approved). Set REDIVIS_API_TOKEN in .env at the repo root.
Never printed, never committed.

## Mechanism note

Redivis' bulk `Table.download_files()` / `Directory.download()` were
observed to silently stop after the first file in this environment (no
exception, exit 0, files short of what was requested) -- a client-side
issue, reproduced independently of network conditions. Per-file
`File.read()` in a plain loop did not exhibit this, so that is the
acquisition primitive here, parallelized with a thread pool for speed
(I/O-bound: ~3.5s/file serially would be ~1.3 hours for both sets).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "data/raw"

DDI_REF = "aimi.ddi_diverse_dermatology_images:3r16:v1_0"
DDI2_REF = "aimi.ddi2_diverse_dermatology_images_2:6f7f:v1_0"

MAX_WORKERS = 8
MAX_RETRIES = 3


def _load_token() -> str:
    token = os.environ.get("REDIVIS_API_TOKEN")
    if token:
        return token

    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("REDIVIS_API_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise SystemExit(
        "REDIVIS_API_TOKEN is not set. Put it in .env as "
        "REDIVIS_API_TOKEN=... (already gitignored)."
    )


def _fetch_one(file_obj, out_dir: Path) -> tuple[str, int, str | None]:
    """Read one file's bytes and write it, with retries on transient errors."""

    dest = out_dir / file_obj.name
    if dest.exists() and dest.stat().st_size > 0:
        return file_obj.name, dest.stat().st_size, None  # already have it

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = file_obj.read()
            dest.write_bytes(data)
            return file_obj.name, len(data), None
        except Exception as error:  # noqa: BLE001 -- reported, not swallowed
            last_error = str(error)
            time.sleep(1.5 * attempt)

    return file_obj.name, 0, last_error


def acquire_images(redivis, table_ref: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = redivis.table(table_ref)
    files = table.list_files()
    print(f"  {table_ref}: {len(files)} files listed")

    done = 0
    errors: list[tuple[str, str]] = []
    start = time.time()

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, f, out_dir): f for f in files}
        for future in cf.as_completed(futures):
            name, size, error = future.result()
            done += 1
            if error:
                errors.append((name, error))
            if done % 50 == 0 or done == len(files):
                elapsed = time.time() - start
                print(f"    {done}/{len(files)}  ({elapsed:.0f}s elapsed, "
                      f"{len(errors)} errors)", flush=True)

    print(f"  done: {done - len(errors)}/{len(files)} written to {out_dir}")
    if errors:
        print(f"  {len(errors)} FAILED after {MAX_RETRIES} retries:")
        for name, error in errors[:10]:
            print(f"    {name}: {error[:100]}")


def acquire_manifest(redivis, dataset_ref: str, table_name_hint: str, out_path: Path) -> None:
    """Save the label/metadata table (not the file blobs) as CSV."""

    ds = redivis.dataset(dataset_ref)
    for table in ds.list_tables():
        if table_name_hint.lower() in table.name.lower():
            df = table.to_pandas_dataframe()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"  manifest: {len(df)} rows -> {out_path}")
            return
    print(f"  WARNING: no table matching {table_name_hint!r} in {dataset_ref}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ddi", action="store_true")
    parser.add_argument("--skip-ddi2", action="store_true")
    parser.add_argument("--images-only", action="store_true")
    parser.add_argument("--manifests-only", action="store_true")
    args = parser.parse_args()

    os.environ["REDIVIS_API_TOKEN"] = _load_token()
    import redivis  # imported after the token is set

    if not args.skip_ddi:
        print("=== DDI ===")
        if not args.images_only:
            acquire_manifest(redivis, DDI_REF, "metadata", OUT_ROOT / "ddi/manifest.csv")
        if not args.manifests_only:
            acquire_images(redivis, f"{DDI_REF}.Files", OUT_ROOT / "ddi/images")

    if not args.skip_ddi2:
        print("=== DDI-2 ===")
        if not args.images_only:
            acquire_manifest(redivis, DDI2_REF, "spreadsheet", OUT_ROOT / "ddi2/manifest.csv")
        if not args.manifests_only:
            acquire_images(redivis, f"{DDI2_REF}.Final DDI2 Asian Photos (no metadata)",
                            OUT_ROOT / "ddi2/images")

    print("\nDone.")


if __name__ == "__main__":
    main()
