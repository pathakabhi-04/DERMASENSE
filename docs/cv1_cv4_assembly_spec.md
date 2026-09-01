# CV-1 → CV-4 Pipeline Assembly — Spec

**Status:** Committed before running, per the same discipline as the
CV-2/CV-3/CV-1.5 specs.

## What this is

Every component through CV-4 exists and is individually validated, but
nothing has ever run them as one pipeline — `src/inference/pipeline.py`
wires only CV-4 → risk. This assembles CV-1 → CV-1.5 → [CV-2] → CV-3 →
CV-4 → risk into one orchestrator and measures the result.

Done before starting CV-5/6/7 deliberately: composing what exists is
cheap and tells us with evidence where the real product-level weakness
is, rather than investing in three unstarted components on an
unverified base.

## Architecture

```
image (BGR ndarray)
  → CV-1  assess_image()   → not usable → STOP  [QUALITY_REJECTED]
  → CV-1.5 route_image()   → "pre_framed" | "wide_field"
       pre_framed → one candidate: full-frame box
       wide_field → CV-2   → 0 boxes → STOP     [NO_CANDIDATES]
                            → N boxes (pixel xyxy)
  → per candidate:
       crop_and_normalize(margin=0.25) → CV-3 tensor + px_box
       CV-3 → mask (EVIDENCE ONLY)
       px_box → RGB crop → 224 transform → CV-4 → diagnosis + action + gate
  → aggregate → PipelineResult                  [ASSESSED]
```

## Committed design decisions

**1. CV-3's mask is evidence; it never touches CV-4's input.**
CV-4 receives the same unmodified crop CV-3 received. The mask is
recorded alongside the diagnosis (area fraction, degenerate flag,
border-touch) for CV-5 explainability and later lesion morphometry.

Reasoning is dependency-direction, not merely the measured ~22% TBP
unreliability: CV-4 drives risk, so it must depend on as few upstream
failure points as possible. Masking or mask-re-cropping would make the
diagnosis conditional on CV-3 being correct, and the failures correlate
in the dangerous direction — a bad mask on a genuinely concerning lesion
could crop away the exact tissue CV-4 needed, converting a CV-3
segmentation miss into a CV-4 false reassurance. That is this product's
worst failure mode; do not create a new path to it absent strong
evidence that masking improves diagnosis (there is none — CV-4 already
works on plain crops). It also preserves independent evaluability of
each component, which has been load-bearing for this project.

**2. Non-assessment is a first-class outcome.**
`PipelineOutcome` ∈ {`QUALITY_REJECTED`, `NO_CANDIDATES`, `ASSESSED`}.
"Never assessed" must be structurally impossible to confuse with
"assessed, low risk" — that is the silent-miss concern encoded in the
type rather than left to convention.

**3. Multi-candidate aggregation is conservative.**
Image-level action = most severe action across candidates
(URGENT_EVALUATION > EVALUATE_SOON > MONITOR > UNKNOWN);
`requires_review` = any candidate requires review. Matches the existing
gate's fail-safe-to-REVIEW philosophy.

**4. Capture guidance is a safety mechanism, and it warns rather than
blocks.** Steering a user toward a close-up routes them onto the only
branch with validated end-to-end evidence (pre-framed = CV-4 on its
validated distribution; wide-field stacks CV-2's ~19% silent miss,
CV-3's ~22% TBP fragmentation, and CV-4 on an unvalidated crop domain).
But a user cannot photograph a mole on their own back or shoulder —
sites where melanoma is most commonly missed in men — so hard-blocking
a wide-field submission would systematically exclude the highest-risk
anatomy. Therefore: suggest, still process, and surface that a wide
submission is a screening pass rather than a full assessment.
Guidance order follows pipeline order: quality issues first, framing
second (never say "move closer" on an image too blurry to interpret).

## Evaluation

**Pre-framed branch — PAD-UFES test (352 images).** Full product metrics
scored against `diagnosis_to_action(native_diagnosis)`, reusing the
existing Tier-1..4 taxonomy verbatim
(`scripts/analyze_phase4_safety_bottleneck.py`, HIGH_RISK = BCC/MEL/SCC).

**This is a regression check, not a new accuracy claim.** Pre-committed
criterion: the assembled pipeline must not degrade CV-4's known
standalone baseline (Macro-F1 0.5996, 32/352 Tier-1 errors). Because the
pre-framed branch feeds CV-4 essentially the whole image, results should
be near-identical; any material deviation IS the finding — most likely a
preprocessing mismatch (cv2 vs PIL resize), worth catching now rather
than after CV-5/6/7 are built on top.

**Wide-field branch — iToBoS test subset.** iToBoS carries no diagnosis
labels, so **no accuracy claim is made**. Report structural propagation
only: quality-rejection rate, routing distribution, zero-candidate rate
(the silent-miss rate), candidates per image, action distribution.

## Anti-rabbit-hole boundary

Assemble, run once per branch, report against the criterion above. No
threshold tuning, no retraining, no chasing metric deltas. If the
regression check fails, diagnose the cause once and fix the mismatch —
that is a bug, not an optimization opportunity.

## Explicitly out of scope

- CV-5/6/7 (the mask is *recorded* for CV-5, not consumed by it here).
- The user-facing suggestion engine's copy, tone, UI, and whether to
  offer "proceed anyway". This task builds the structured signal layer
  only; presentation is a separate product decision.
- Rewiring `src/inference/pipeline.py`. It stays as the CV-4-only path
  (three existing tests depend on its `predict(tensor)` contract); the
  assembled pipeline is a new class alongside it.
