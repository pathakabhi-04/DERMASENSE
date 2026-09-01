# CV-6 Uncertainty — Spec

**Status:** Committed before implementation, per the same discipline as
prior specs.

## What CV-6 is

Confidence and abstention signal, sitting after CV-4 alongside CV-5/CV-7,
feeding CV-8 convergently (`docs/project_state.md`). Directly motivated
by a gap left open in the CV-4 domain-evidence work: `crop_contrast`
(`docs/cv4_domain_evidence_spec.md`) was added as a disclosure signal
correlating with BCC/ACK's out-of-domain confidence collapse, but
nothing yet consumes it or complements it with an independent
uncertainty estimate.

## Design principle: evidence, not a decision

CV-6 must NOT modify `src/risk/safety_gate.py`. Confirmed by direct
inspection: that module is currently pure diagnosis-string lookup
(`evaluate_prediction(predicted_diagnosis: str)`), with zero confidence
parameter anywhere in its signatures or logic. It stays that way here.

This follows the same reasoning already applied twice this session: CV-3's
mask is evidence, never gating CV-4's input (dependency-direction
argument — CV-4 drives risk, so it depends on as few upstream failure
points as possible); `crop_contrast` is evidence, never filtering
candidates (would recreate the silent-miss failure mode). Collapsing
multiple uncertainty signals into one actionable decision is CV-8's job
— the convergent risk engine that receives CV-4 + CV-5 + CV-6 + CV-7
together, per the locked architecture. Building that collapse now, one
component early, would be the same premature-scope-grab this project has
twice already declined to make.

## Three evidence signals, all zero-new-training

**1. Ensemble disagreement.** Two independently-trained checkpoints
already exist — `checkpoints/archive/pad_ufes_c1_partial_finetune_seed{42,123}_best.pt`
— loadable via the existing `NativePredictor.from_checkpoint` with no
new loading code. Run both per candidate; record whether their
predicted classes agree, and the L1 distance between their probability
vectors. This is the cheapest possible version of a well-established
uncertainty proxy, since the second model is sitting on disk unused.

**2. Temperature-calibrated confidence.** `compute_ece` currently lives
only in `scripts/evaluate_c1_vs_f1_product.py` (pure function, no `src/`
dependency) — moved into `src/uncertainty/calibration.py`. A single
scalar temperature T is fit on PAD-UFES val (labeled, already available
locally) by grid search minimizing ECE. Applied strictly post-hoc, on
probabilities only: `calibrated = softmax(log(p)/T)`. This is
mathematically equivalent to true logit-temperature-scaling — the
unknown softmax normalization constant cancels in the ratio — so it
requires **no change to `NativePredictor`/`native.py`**, and no logits
need to be exposed. Both raw and calibrated confidence are recorded.

**3. `crop_contrast`/`crop_blur`.** Already computed, already on
`CandidateResult` (`docs/cv4_domain_evidence_spec.md`). No new code —
documented here as part of CV-6's evidence set, since it is exactly this
kind of signal and was added anticipating this use.

None of the three requires new training, new labeled data, or a change
to CV-4's weights.

## Explicitly deferred, not attempted

- **MC-Dropout.** `NativeClassifierConfig.dropout` is architecturally
  wired but hardcoded to 0.0 at load time
  (`NativePredictor.from_checkpoint`), and the current checkpoint was
  not trained with dropout active. Enabling it needs a retrained,
  dropout-compatible checkpoint plus non-standard train-mode-for-dropout-
  only inference handling — real cost, not in scope here.
- **Logit-based OOD scoring (max-logit, energy score).** Would require
  exposing raw logits from `NativePredictor.predict()`, a more invasive
  change than temperature scaling needs, for marginal value over the
  three signals above. Deferred, not ruled out permanently.

## Pre-committed evaluation

Re-run `scripts/evaluate_pipeline_end_to_end.py` on both branches
(PAD-UFES full test set, iToBoS the same 1,000-image seeded sample used
throughout this branch's evaluation) with the new fields populated.

**The specific question this answers:** does ensemble disagreement
independently spike on the same classes (BCC, ACK) where `crop_contrast`
already showed a collapse? This is not an invented metric — it is a
direct corroboration check against an existing, already-documented
finding.

- **If yes:** two independent signals (one from input-crop texture, one
  from model disagreement) corroborate the same out-of-domain weak
  spot — meaningfully stronger evidence than either alone, and worth
  noting as such.
- **If no:** informative on its own terms — it would mean the two
  signals capture different failure modes, not that either is wrong.
  Documented either way, not chased into a third signal to force
  agreement.

**No pass/fail gate.** Like the CV-4 domain-evidence work, this is an
evidence-layer addition, not a component with an accuracy target. The
classification, action, and gate outputs are unchanged by design.

## Anti-rabbit-hole boundary

Three signals, computed once, evaluated once against the corroboration
question above. Do not add MC-Dropout, logit-based OOD, or a fourth
uncertainty method chasing a stronger correlation. Do not build an
abstention policy or touch `safety_gate.py` — that is out of scope for
CV-6 by design, reserved for CV-8.
