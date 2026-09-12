"""
Acquire a bounded sample of Fitzpatrick17k to the external drive.

Why this specific second source: it is deliberately NOT another Stanford/
Redivis dataset. DDI + DDI-2 (referral_head_on_ddi.md,
native_classifier_on_ddi.md) showed the referral head's and the native
classifier's malignant-vs-benign discrimination both collapse on clinical
photography off ISIC/PAD-UFES -- but that rests on one institution's
patient photos. Fitzpatrick17k is atlas-scraped teaching images from two
independent dermatology reference sites (dermaamin.com,
atlasdermatologico.com.br), a different capture/curation pipeline
entirely. If the collapse reproduces here too, it is evidence about
clinical photography generally, not an artifact of DDI specifically.

Source: https://github.com/mattgroh/fitzpatrick17k (fitzpatrick17k.csv),
public, no data-use agreement. `three_partition_label` already gives
malignant/benign/non-neoplastic; `label` carries the fine-grained
diagnosis (contains "melanoma" for melanoma cases).

## Bounded, not exhaustive

The full CSV has 16,577 rows across two atlas sites with a documented,
nontrivial broken-link rate (each image is fetched from the ORIGINAL
atlas URL, not a stable archive). Downloading everything would not be
"hours, no GPU" and is not needed for a directional check: this samples
up to MAX_MELANOMA melanoma-labelled rows (the headline metric) and a
comparable pool of other-malignant and benign rows, roughly matching
DDI+DDI-2's own scale (~1,300 attempted total), and reports the actual
usable count plainly rather than assuming the sample lands intact.

    python -m scripts.acquire_fitzpatrick17k
"""

from __future__ import annotations

import concurrent.futures as cf
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data/raw/fitzpatrick17k"
CSV_URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"

RANDOM_SEED = 20260913  # reproducible sample, tied to the date this ran
MAX_MELANOMA = 250
MAX_OTHER_MALIGNANT = 200
MAX_BENIGN = 300

# dermaamin.com (12,631 of 16,577 rows) is dead: verified 0/5 real image
# URLs resolve (404), despite the site root itself returning 200 -- the
# old /clinical-pic/ path structure this CSV was captured against no
# longer exists. atlasdermatologico.com.br (3,905 rows) is unaffected:
# 6/6 test URLs resolved. Restricting acquisition to the live domain
# rather than spending time on ~12.6k links already confirmed dead.
LIVE_DOMAIN = "atlasdermatologico.com.br"

MAX_WORKERS = 12
TIMEOUT_S = 8
MAX_RETRIES = 2


def _fetch_one(row: dict, out_dir: Path) -> tuple[str, bool, str | None]:
    """Download one image by its md5hash-derived filename. Failure is
    expected and tolerated here -- these are 5-15 year old links to two
    external atlas sites, not a maintained archive."""

    filename = f"{row['md5hash']}.jpg"
    dest = out_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return filename, True, None

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(row["url"], timeout=TIMEOUT_S,
                                     headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            if len(response.content) < 1000:  # smaller than any real photo
                raise ValueError(f"suspiciously small response ({len(response.content)}b)")
            dest.write_bytes(response.content)
            return filename, True, None
        except Exception as error:  # noqa: BLE001 -- reported, tolerated
            last_error = str(error)
            time.sleep(0.5 * attempt)

    return filename, False, last_error


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "fitzpatrick17k_full.csv"

    print("Fetching fitzpatrick17k.csv...")
    response = requests.get(CSV_URL, timeout=30)
    response.raise_for_status()
    csv_path.write_bytes(response.content)

    df = pd.read_csv(csv_path)
    print(f"  {len(df)} total rows")

    df = df[df["url"].str.contains(LIVE_DOMAIN, na=False)].copy()
    print(f"  {len(df)} rows on the live domain ({LIVE_DOMAIN}); "
          f"dropped {16577 - len(df)} on the confirmed-dead dermaamin.com")

    is_mel = df["label"].str.contains("melanoma", case=False, na=False)
    malignant = df["three_partition_label"] == "malignant"
    benign = df["three_partition_label"] == "benign"

    mel_sample = df[is_mel].sample(
        n=min(MAX_MELANOMA, is_mel.sum()), random_state=RANDOM_SEED
    )
    other_mal_sample = df[malignant & ~is_mel].sample(
        n=min(MAX_OTHER_MALIGNANT, (malignant & ~is_mel).sum()), random_state=RANDOM_SEED
    )
    benign_sample = df[benign].sample(
        n=min(MAX_BENIGN, benign.sum()), random_state=RANDOM_SEED
    )

    sample = pd.concat([mel_sample, other_mal_sample, benign_sample], ignore_index=True)
    print(f"  sampled: {len(mel_sample)} melanoma, {len(other_mal_sample)} other-malignant, "
          f"{len(benign_sample)} benign  (n={len(sample)})")

    images_dir = OUT_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    done = 0
    failures: list[tuple[str, str]] = []
    filenames: list[str] = []
    start = time.time()

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, row.to_dict(), images_dir): row.to_dict()
            for _, row in sample.iterrows()
        }
        for future in cf.as_completed(futures):
            filename, ok, error = future.result()
            done += 1
            filenames.append(filename if ok else None)
            if not ok:
                failures.append((filename, error))
            if done % 100 == 0 or done == len(sample):
                elapsed = time.time() - start
                print(f"    {done}/{len(sample)}  ({elapsed:.0f}s, "
                      f"{len(failures)} failed so far)", flush=True)

    sample = sample.reset_index(drop=True)
    sample["filename"] = filenames
    sample["downloaded"] = sample["filename"].notna()

    manifest_path = OUT_DIR / "manifest.csv"
    sample.to_csv(manifest_path, index=False)

    n_ok = sample["downloaded"].sum()
    n_mel_ok = sample.loc[sample["downloaded"], "label"].str.contains(
        "melanoma", case=False, na=False
    ).sum()

    print(f"\n  downloaded: {n_ok}/{len(sample)}  ({100*n_ok/len(sample):.0f}%)")
    print(f"  usable melanoma images: {n_mel_ok}")
    print(f"  manifest written to {manifest_path}")
    if failures:
        print(f"\n  {len(failures)} failures (expected for old atlas links), examples:")
        for name, error in failures[:5]:
            print(f"    {name}: {error[:90]}")


if __name__ == "__main__":
    main()
