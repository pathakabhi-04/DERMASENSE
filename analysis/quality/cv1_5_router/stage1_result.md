# CV-1.5 Domain Router — Stage 1 (Heuristic) Result

**Status:** FAIL against the pre-committed gate. Escalation to Stage 2
(learned classifier) is the indicated next step per
`docs/cv1_5_router_spec.md`, but that requires a training run (likely
RunPod GPU) — not started without explicit go-ahead.

## Setup
Per `docs/cv1_5_router_spec.md`. `src/routing/heuristic.py` (pigmentation-
contrast blob analysis: largest-blob-area-fraction of frame, HSV
saturation/value thresholding relative to each image's own median tone).
Threshold calibrated once via a train-split sweep (200 images, separate
from the held-out set) — see script/module docstrings; blob COUNT
calibrated out as a discriminator (pre_framed images showed MORE small
blobs than wide_field, opposite the hypothesized direction — closer-zoom
clinical photos resolve more visible texture, freckles, hair, marker
dots, not fewer).

Held-out evaluation: 150 PAD-UFES-20 test images (label `pre_framed`) +
150 iToBoS test images (label `wide_field`), seed=42, proxy-labeled by
dataset identity (see spec's ground-truth caveat).

## Result

| metric | value | gate |
|---|---|---|
| pre_framed accuracy | 0.800 | >= 0.90 |
| wide_field accuracy | 0.620 | >= 0.90 |
| overall accuracy | 0.710 | (not gating) |

Confusion matrix:

| true \\ predicted | pre_framed | wide_field |
|---|---|---|
| pre_framed | 120 | 30 |
| wide_field | 57 | 93 |

Both classes fail the gate, wide_field substantially so (38% of true
wide-field images get misrouted as pre_framed). Consistent with the
calibration-sample ceiling (~75% balanced accuracy) found before this
run — not a surprise, not evidence of overfitting either way.

## Why this feature isn't enough
A single largest-pigmented-blob-fraction signal conflates two things
that don't line up cleanly: framing (how close the camera is) and
lesion salience (how visually distinct the lesion is from surrounding
skin). Some iToBoS wide-field photos still contain a visually prominent
lesion or dark mole that trips the blob threshold; some PAD-UFES
close-ups have low-contrast/faint lesions that don't. A single classical
feature does not separate these reliably — same category of finding as
CV-2 needing a trained detector rather than classical CV for wide-field
localization (`docs/cv2_detection_spec.md` Section 2.1).

## Decision (per docs/cv1_5_router_spec.md)

Per the pre-committed decision rule: Stage 1 fails → escalate to Stage 2
(lightweight learned classifier, ResNet18 fine-tune on PAD-UFES vs.
iToBoS train splits). This is the point flagged in the spec as worth a
RunPod GPU session — not started here; needs explicit sign-off before
spinning up a pod and a training run, per the plan.

No further heuristic-threshold iteration was attempted — per the spec's
anti-rabbit-hole boundary, one heuristic attempt, evaluated once, is the
bound.
