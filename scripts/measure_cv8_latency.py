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

Reports cold (first call, includes lazy init) separately from warm,
because only warm latency describes steady-state serving.

    python -m scripts.measure_cv8_latency
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import cv2
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

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  cold (first call) : {cold:.2f}s")
    print(f"  warm n            : {len(warm)}")
    print(f"  warm min / p50 / max: {warm_sorted[0]:.2f}s / {p50:.2f}s / "
          f"{warm_sorted[-1]:.2f}s")
    print(f"  warm mean         : {statistics.mean(warm):.2f}s")

    print(f"\n  Pre-committed rule: p50 < {DECISION_THRESHOLD_S}s -> build sync endpoint")
    if p50 < DECISION_THRESHOLD_S:
        print(f"  DECISION: p50 {p50:.2f}s < {DECISION_THRESHOLD_S}s -> "
              "SYNC endpoint is viable.")
    else:
        print(f"  DECISION: p50 {p50:.2f}s >= {DECISION_THRESHOLD_S}s -> "
              "DO NOT build the sync endpoint yet.")
        print("            Take sync-vs-async back to the RAG side first.")
    print("\n  Caveat: CPU only, and this run has no prior image, so CV-7 does")
    print("  not execute. A returning-visit request segments TWO images and")
    print("  will cost more than the figure above.")


if __name__ == "__main__":
    main()
