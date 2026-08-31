# CV-1.5 Domain Router — Result

**Status:** PASS (Stage 2). CV-1.5 is a ResNet18 fine-tune, not the
Stage 1 heuristic. `src/routing/classifier.py` +
`checkpoints/cv1_5_router/best.pt` (RunPod volume, not in git — model
weights are never committed per project convention).

## Stage 1 (heuristic) — FAILED

80.0% pre_framed / 62.0% wide_field vs. the >=90%-per-class gate. Full
write-up (confusion matrix, failure-mode analysis): `stage1_result.md`.
Superseded below, kept for the record per the spec's staged-escalation
design.

## Stage 2 (ResNet18 classifier) — PASS

Trained on RunPod GPU (`scripts/train_cv1_5_router.py`, PAD-UFES train
split -> pre_framed, iToBoS train split -> wide_field, ImageNet-
pretrained backbone, 8 epochs). Scored via
`scripts/evaluate_cv1_5_router.py` against the exact same 150+150
held-out set Stage 1 used (`analysis/quality/cv1_5_router/eval_set.csv`,
seed=42 — never retrained or resampled after seeing Stage 1's result).

| metric | value | gate |
|---|---|---|
| pre_framed accuracy | 1.000 | >= 0.90 |
| wide_field accuracy | 1.000 | >= 0.90 |

300/300 correct, both classes perfect (`stage2_predictions.csv`).

## Why a perfect score isn't treated as suspicious here

A 100%/100% result on a proxy-labeled task (label = dataset identity,
not per-image-verified framing — see `docs/cv1_5_router_spec.md`'s
ground-truth caveat) deserves scrutiny before being accepted at face
value: a classifier can shortcut on dataset fingerprints (compression,
color calibration, camera signature) rather than learning genuine
framing semantics, and a perfect score is if anything weaker evidence of
robustness than a clean 92-95% would be.

Bounded sanity check performed before accepting the result: visually
inspected 6 eval-set images (3 pre_framed, 3 wide_field) directly. They
are genuinely, obviously different at the composition level — pre_framed
images are tight macro shots of one lesion filling most of the frame;
wide_field images are much wider body-region photos (a whole foot, a
large patch of skin, a hairy limb) with no dominant close-up lesion. Not
a labeling bug, not corrupted images. This also explains why Stage 2
succeeded where Stage 1 didn't: a pretrained CNN captures holistic
scene/framing composition (image scale, background context, body
landmarks), which is a fundamentally richer signal than Stage 1's single
pigmentation-blob-salience heuristic.

## What still stands as a caveat (not a blocker)

The proxy-label gap remains real: this measures "PAD-UFES vs. iToBoS,"
not verified per-image framing. Real user-submitted photos won't always
match either curated dataset's capture style as cleanly as this
held-out set does. If CV-1.5 underperforms in production, the proxy-vs-
true-label gap is the first place to look — not evidence the method is
broken. Not re-litigated further here per the spec's anti-rabbit-hole
boundary; flagged for whenever CV-1.5 gets evaluated against real
product images (same category of deferred validation as CV-1's
"synthetic-only" caveat and CV-2's "iToBoS->phone, needs real phone-image
test set" caveat).

## Decision

Per `docs/cv1_5_router_spec.md`: Stage 2 passes both per-class gates ->
CV-1.5 is done. `src/routing/classifier.py::route_image` (+
`load_router_checkpoint`) is the production interface. No further
iteration (architecture search, threshold tuning, resampling) per the
anti-rabbit-hole boundary — this closes CV-1.5.

**Explicitly still out of scope:** wiring this into
`src/inference/pipeline.py` as part of an actual CV-1->CV-8 orchestrator.
That was scoped out of this task from the start (see the plan) and
remains a separate follow-on step.
