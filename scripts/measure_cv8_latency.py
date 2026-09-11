"""
Measure `DermaSensePipeline.predict()` wall-clock latency.

The one measurement gating the live-feed sync/async decision
(docs/build_on_baseline_1.md Section A, which defers to Section B item
14). Pre-committed decision rule, fixed BEFORE running:

    p50 < 2.0s  -> build the minimal synchronous HTTP endpoint that
                   Section A already describes as the default.
    p50 >= 2.0s -> do not build it; take the sync-vs-async question back
                   to the RAG collaborator first, because a multi-second
                   blocking call changes what their chat UI can do.

Measures BOTH request shapes, because they differ by ~5x and only one
of them was measured the first time this ran:

- first visit     -- one image, CV-7 does not execute
- returning visit -- prior image supplied, so TWO images are segmented
                     and CV-7 runs

The first run of this script measured only first visits, reported
p50 0.52s, and concluded sync was viable. That conclusion did not hold:
returning visits measure ~2.45s and fail the same rule. Measuring only
the cheap path is how a latency budget gets set on the wrong number, so
the returning-visit case is no longer optional here.

Reports cold (first call, includes lazy init) separately from warm,
because only warm latency describes steady-state serving.

    python -m scripts.measure_cv8_latency
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.inference.orchestrator import DermaSensePipeline, PipelineOutcome

REPO_ROOT = Path(__file__).resolve().parents[1]
PAD_UFES_TEST = REPO_ROOT / "data/splits/pad_ufes/test.csv"
ROUTER_CHECKPOINT = REPO_ROOT / "checkpoints/cv1_5_router/best.pt"
SEGMENTATION_CHECKPOINT = REPO_ROOT / "checkpoints/cv3_512/best.pt"
CLASSIFIER_CHECKPOINT = (
    REPO_ROOT / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
)

DECISION_THRESHOLD_S = 2.0
WARM_RUNS = 8


def _measure_returning_visits(pipeline) -> list[float]:
    """
    Time the prior-image path, which segments two images and runs CV-7.

    Uses the same real visit pairs as the delivered fixtures, so the
    figure describes the request shape the RAG side will actually issue
    for a returning patient.
    """

    import json
    import zipfile

    fixtures = REPO_ROOT / "docs/cv8_sample_outputs/sample_outputs.json"
    archive = REPO_ROOT / "data/raw/UQ_zip/866990d01449152d_NIMARE-A11453_A11453.zip"

    if not fixtures.exists() or not archive.exists():
        return []

    pairs = [
        entry["source"]
        for entry in json.loads(fixtures.read_text())
        if isinstance(entry.get("source"), dict)
    ]
    if not pairs:
        return []

    print(f"\nMeasuring returning visits (prior image supplied), "
          f"{len(pairs)} real pairs...")

    timings: list[float] = []
    with zipfile.ZipFile(archive) as zf:
        for index, source in enumerate(pairs):
            earlier = cv2.imdecode(
                np.frombuffer(zf.read(source["earlier"]), np.uint8), cv2.IMREAD_COLOR)
            later = cv2.imdecode(
                np.frombuffer(zf.read(source["later"]), np.uint8), cv2.IMREAD_COLOR)

            start = time.perf_counter()
            pipeline.predict(later, lesion_id=f"ret-{index}", prior_image_bgr=earlier)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            print(f"  [ret ] {elapsed:6.2f}s")

    return timings


def main() -> None:
    print("Loading checkpoints (not timed -- a real service loads these once "
          "at startup, never per request)...")
    load_start = time.perf_counter()
    pipeline = DermaSensePipeline.from_checkpoints(
        router_checkpoint=ROUTER_CHECKPOINT,
        segmentation_checkpoint=SEGMENTATION_CHECKPOINT,
        classifier_checkpoint=CLASSIFIER_CHECKPOINT,
        detector_weights=None,
        device="cpu",
    )
    print(f"  startup cost: {time.perf_counter() - load_start:.1f}s")

    rows = pd.read_csv(PAD_UFES_TEST)
    images = []
    for _, row in rows.head(WARM_RUNS + 1).iterrows():
        img = cv2.imread(str(REPO_ROOT / row["image_path"]))
        if img is not None:
            images.append((row["image_path"], img))

    if len(images) < 2:
        raise SystemExit("Could not read enough test images (is the HDD attached?)")

    print(f"\nMeasuring predict() on {len(images)} real PAD-UFES images, CPU...")

    timings: list[float] = []
    for index, (path, img) in enumerate(images):
        start = time.perf_counter()
        result = pipeline.predict(img, lesion_id=f"latency-{index}")
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        label = "cold" if index == 0 else "warm"
        print(f"  [{label:4}] {elapsed:6.2f}s  outcome={result.outcome.value}")

    cold, warm = timings[0], timings[1:]
    warm_sorted = sorted(warm)
    p50 = statistics.median(warm)

    # Returning visit: two segmentations plus CV-7.
    returning = _measure_returning_visits(pipeline)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  cold (first call) : {cold:.2f}s")
    print(f"  warm n            : {len(warm)}")
    print(f"  warm min / p50 / max: {warm_sorted[0]:.2f}s / {p50:.2f}s / "
          f"{warm_sorted[-1]:.2f}s")
    print(f"  warm mean         : {statistics.mean(warm):.2f}s")

    if returning:
        r_sorted = sorted(returning)
        r_p50 = statistics.median(returning)
        print(f"\n  returning visit n  : {len(returning)}")
        print(f"  returning min/p50/max: {r_sorted[0]:.2f}s / {r_p50:.2f}s / "
              f"{r_sorted[-1]:.2f}s   ({r_p50 / p50:.1f}x first visit)")
    else:
        r_p50 = None
        print("\n  returning visit    : not measured (UQ sample unavailable)")

    print(f"\n  Pre-committed rule: p50 < {DECISION_THRESHOLD_S}s -> build sync endpoint")
    print(f"  first visit     {p50:.2f}s  "
          f"{'PASSES' if p50 < DECISION_THRESHOLD_S else 'FAILS'}")
    if r_p50 is not None:
        print(f"  returning visit {r_p50:.2f}s  "
              f"{'PASSES' if r_p50 < DECISION_THRESHOLD_S else 'FAILS'}")

    worst = max(p50, r_p50) if r_p50 is not None else p50
    if worst < DECISION_THRESHOLD_S:
        print("\n  DECISION: sync endpoint is viable for both request shapes.")
    else:
        print(f"\n  DECISION: the worst shape is {worst:.2f}s, over the "
              f"{DECISION_THRESHOLD_S}s bar.")
        print("            Sync is NOT settled -- take sync-vs-async back to")
        print("            the RAG side, per the rule set before measuring.")
    print("\n  Caveat: CPU only. A GPU would likely change both figures, but")
    print("  that is untested and must not be assumed.")


if __name__ == "__main__":
    main()
