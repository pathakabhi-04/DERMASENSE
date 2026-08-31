# CV-2 Experiment E — Result: YOLO11s Capacity Test

**Status:** Negative result. Increased model capacity did not move the
image-level recall gate. Confirms the CV-2 recall ceiling is not a capacity
problem.

## Hypothesis
The ~19% of lesion-containing images where B1 (YOLO11n) surfaces no
candidate were hypothesized to be missed because a nano-capacity model
(~2.6M params) lacks the representational capacity to reliably fire on
small/faint lesions at this dataset's extreme small-object scale (median
lesion normalized area 0.00097). A larger backbone (YOLO11s, ~9.4M params)
was expected to recover some of these and lift image-level recall toward
the committed >= 0.90 gate.

## Setup
- Single-variable change from B1: backbone only (yolo11n.pt -> yolo11s.pt).
- Original train.txt (NOT D's oversampled list) — capacity tested in
  isolation.
- imgsz 1280, epochs 50, batch 8, seed 42 — all identical to B1.
- Trained on RTX 4090, torch 2.8.0+cu128. (Note: environment issues this
  session — the network-volume .venv was broken; training ran on the system
  Python interpreter. Does not affect result validity.)

## Result (val split, conf=0.25, revised metrics)
| Metric                  | B1 (nano) | E (YOLO11s) | Change   | Gate     |
|-------------------------|-----------|-------------|----------|----------|
| Image-level recall      | 0.8089    | 0.8098      | +0.0009  | >= 0.90  |
| Zero-lesion FP burden median | 0.00 | 0.00        | —        | <= 1     |
| Zero-lesion FP burden p90    | 1.00 | 1.00        | —        | <= 2     |
| Zero-lesion FP burden max    | 10   | 12          | +2       | (watch)  |
| Binary zero-lesion FPR (reported) | 0.2215 | 0.1758 | -0.0457 | (reported) |

## Conclusion
Image-level recall moved +0.0009 — pure noise. A 3.6x larger model surfaced
a lesion in essentially the identical set of images. The ~19% complete-miss
rate is NOT a capacity problem.

Note the conf=0.001 diagnostic (both B1 and E) shows the model *finds* ~96%
of lesions when allowed to — the missed lesions are not invisible to the
detector, they score below usable confidence at whole-image scale. This is
consistent with an object-scale problem rather than a capacity problem, and
is what makes tiling/SAHI (which enlarges small objects per-tile) the one
mechanistically-motivated remaining lever.

Binary FPR improved (0.22 -> 0.18) but this is a reported secondary, not a
gate; the burden gates (the actual FP measure) were already met by B1 and
are unchanged.

## Decision (per Section 22 committed stopping rule)
Recall barely moved from B1 -> **not a capacity problem**. Per the committed
rule: do NOT escalate to YOLO11m/l/x. Capacity escalation is closed.

CV-2 is accepted as **baseline at the 0.81 image-level recall floor**, with
refinement deferred (see cv2_status.md). The one remaining permitted
experiment (tiling/SAHI) is deferred, not abandoned — its re-entry is gated
on end-to-end pipeline evaluation showing CV-2's miss rate is a dominant
failure mode.

## Experiments run against the CV-2 recall/FPR problem (complete record)
1. Confidence threshold sweep — no usable operating point exists.
2. B0 -> B1 resolution (1024 -> 1280) — no effect on locked metrics.
3. D — sun-damage hard-negative oversampling (8x) — no effect.
4. E — YOLO11s capacity — no effect on recall.

Four levers tried; the recall ceiling held at ~0.81 image-level across all.
The remaining untried lever is object-scale (tiling), deferred per above.