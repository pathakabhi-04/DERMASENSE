# CV-1.5 Domain Router — Spec

**Status:** Committed before running (pre-registered criteria), per the
same discipline as CV-2/CV-3 experiment specs.

## What this validates

CV-1.5 decides, per input image, whether it is **pre-framed /
lesion-centric** (route straight to CV-3) or **wide-field** (route
through CV-2 detection first, then crop into CV-3). This is a locked
architecture decision (`docs/project_state.md`); what's undecided is the
mechanism. This spec covers Stage 1 (a classical, no-training heuristic)
and pre-commits the escalation criterion to Stage 2 (a learned
classifier) so that decision isn't made post-hoc after seeing results.

## Ground truth caveat (read before interpreting any result)

Neither PAD-UFES-20 nor iToBoS 2024 metadata has a per-image framing
label. The label used here is a **proxy: dataset identity**.
PAD-UFES = framed (asserted "visually confirmed" elsewhere in this
project, not per-image verified). iToBoS = wide-field (by construction —
it's the wide-field TBP detection dataset). Every accuracy number below
therefore measures "can this distinguish a PAD-UFES-style photo from an
iToBoS-style photo," which is a reasonable stand-in for the true framing
question but is not identical to it. If CV-1.5 is ever evaluated against
real user-submitted photos and underperforms this benchmark, the gap
between proxy-label and true-framing-label is the first place to look —
not a sign the method is broken.

## Held-out evaluation set

Fixed, seeded sample (seed=42), built once and reused for both stages:
- n=150 from `data/splits/pad_ufes/test.csv` → label `pre_framed`
- n=150 from `data/splits/itobos_detection/test.csv`, deduplicated to
  unique `image_id` (8,481 available) → label `wide_field`

Larger than the CV-3 domain-audit sample (n=50) deliberately: a routing
mistake here changes which pipeline branch an image takes — a
structural error, not a quality-degradation proxy — so the bar for
confidence in the number is higher.

## Pre-committed gate

**≥ 90% accuracy on EACH class separately** (not just overall — a
router that's 95% on framed and 60% on wide-field averages to a
misleadingly comfortable 77%, so both `pre_framed` recall and
`wide_field` recall must individually clear 90%).

This gate is stricter than CV-2's 0.90 target-but-0.81-accepted-floor or
CV-3's 0.75 Dice floor, deliberately: those measure degradation of a
downstream quality signal; this measures whether the pipeline sends an
image down the structurally correct branch at all.

## Stage 1 — classical heuristic (run first)

`src/routing/heuristic.py`. No training, no ground-truth masks needed.
Signal: does the frame contain one dominant foreground blob filling most
of it (PAD-UFES-style close-up clinical photo) vs. a large area of
relatively uniform skin with no single dominant blob (iToBoS-style
wide-field photo)? Implementation: skin/lesion-toned region via HSV
saturation+value thresholding, largest-connected-component area as a
fraction of the frame, plus blob count above a minimum size. Threshold
values are initial engineering defaults (same caveat CV-1's signals
carry) — they are tuned once against the held-out set below, not
iteratively chased.

## Decision rule

- **Both classes ≥ 90% on the held-out set → PASS.** Stage 1 heuristic
  is CV-1.5. No training, no GPU needed. Document result, stop.
- **Either class < 90% → Stage 2.** Escalate to a lightweight learned
  classifier (`src/routing/classifier.py`,
  `scripts/train_cv1_5_router.py`, ResNet18 fine-tune on the PAD-UFES /
  iToBoS train splits, same proxy-label caveat applies to training data
  too). This is the point where a RunPod GPU session is worth spinning
  up — flag to the user before starting the training run, don't just
  start it. Evaluate Stage 2 against the SAME held-out set and the same
  per-class ≥90% gate.
- **If Stage 2 also fails the gate:** document as a known limitation
  (per-class accuracy numbers, dominant confusion direction) and ship
  with the best available option rather than continuing to iterate. A
  misroute here is not catastrophic by pipeline design — a wide-field
  image processed lesion-centric or vice versa degrades quality (same
  category as CV-2's known ~19% miss rate, CV-3's ~22% TBP fragmentation
  rate) but does not crash the system. Fine-tuning a whole third
  approach is out of scope for this task.

## Anti-rabbit-hole boundary

One heuristic attempt, evaluated once against the fixed held-out set. If
it fails, one classifier attempt, evaluated once against the same set.
Do not iterate thresholds or architectures chasing marginal accuracy
beyond these two attempts, and do not enlarge or resample the held-out
set after seeing a result that's close to the gate. Apply the decision
rule above and stop.
