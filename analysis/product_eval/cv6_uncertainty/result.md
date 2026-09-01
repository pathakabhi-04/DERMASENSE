# CV-6 Uncertainty — Result

**Status:** Implemented (ensemble disagreement, temperature calibration,
`crop_contrast`/`crop_blur` documented as shared evidence). No pass/fail
gate — evidence-layer addition, per `docs/cv6_uncertainty_spec.md`.

## What was built

- `src/uncertainty/calibration.py` — `expected_calibration_error`
  (moved/generalized from `scripts/evaluate_c1_vs_f1_product.py`,
  logic unchanged), `apply_temperature` (post-hoc, probabilities only —
  `softmax(log(p)/T)`, mathematically equivalent to logit-temperature-
  scaling), `fit_temperature` (grid search).
- `src/uncertainty/ensemble.py` — `load_ensemble`, `ensemble_evidence`
  (agreement, mean pairwise probability L1 distance, confidence spread).
- `src/inference/orchestrator.py` — `CandidateResult` gained
  `calibrated_confidence`, `ensemble_agree`, `ensemble_probability_distance`,
  `ensemble_confidence_spread`. Ensemble is opt-in
  (`additional_ensemble_checkpoints`, `--ensemble` flag on the eval
  script) — off by default since it roughly doubles CV-4 inference cost
  per candidate; calibration is always-on (cheap, no extra inference).
- Temperature fit once via `scripts/calibrate_cv6_temperature.py` on
  PAD-UFES val (n=336): raw ECE 0.0596 → calibrated ECE 0.0401 at
  T=1.25. Model is mildly overconfident; calibration measurably helps.
  Stored as `DEFAULT_TEMPERATURE`, not refit at pipeline construction.

## Regression check (PAD-UFES, 352 images, ensemble enabled)

Identical to the pre-CV-6 result: 100% per-image agreement with CV-4
standalone, Macro-F1 0.6521, 31 Tier-1 errors, 346/352 assessed. Adding
CV-6 evidence changes nothing about the primary classification, action,
or gate output — confirms the "evidence, not a decision" design held in
practice, not just on paper.

## The corroboration question (1,000-image iToBoS sample, 3,336 candidates)

**Does ensemble disagreement independently spike on BCC/ACK, the classes
`crop_contrast` flagged? Partially, and the exceptions are the
interesting part.**

| class | n | confidence | crop_contrast | ensemble agree rate | ensemble prob. distance |
|---|---|---|---|---|---|
| NEV | 1808 | 0.808 | 0.303 | 0.997 | 0.083 |
| ACK | 244 | 0.604 | 0.149 | 0.918 | 0.177 |
| SEK | 838 | 0.666 | 0.204 | 0.869 | 0.196 |
| BCC | 403 | 0.576 | 0.207 | 0.792 | 0.201 |
| MEL | 42 | 0.670 | 0.381 | 0.833 | 0.266 |

Overall correlation between `crop_contrast` and ensemble probability
distance across all 3,336 candidates: **r = −0.259, p < 0.0001** — real,
but modest. Confidence itself is a stronger correlate of disagreement
(r = −0.573), which is close to definitional (both partly reflect
general prediction certainty) and not itself informative about the
out-of-domain question.

**BCC replicates the pattern** — third-lowest agreement rate (0.792),
consistent with `crop_contrast`'s finding and with the already-documented
SCC/BCC representation overlap (`analysis/scc_bcc/`): moderate ensemble
disagreement on BCC is expected from a component known to sit in a
confusable region of the embedding space.

**ACK does NOT show the disagreement its crop_contrast collapse would
predict** — 0.918 agreement rate, second-highest of all six classes,
despite having the worst crop_contrast (0.149, 88.1% of ACK crops
below the 0.20 threshold). Both ensemble members agree confidently on
the same low-information crops. This is more concerning than
disagreement would be: a genuinely uncertain model at least signals its
uncertainty via disagreement; two models confidently agreeing on a
diffuse, low-information crop suggests both learned the same spurious
correlation (crop texture, not lesion morphology) rather than either
being unsure. Ensemble disagreement alone would miss this failure
entirely — it needs `crop_contrast` to catch it.

**MEL shows the highest disagreement of any real class** (0.266 mean
distance, 0.833 agreement) despite the visual audit
(`analysis/product_eval/cv4_domain_evidence/result.md`) confirming MEL
crops are consistently coherent, real-looking pigmented lesions. This
is not a contradiction — it is a different failure mode entirely:
MEL/NEV is a classically hard boundary in dermatology (a dark,
irregular mole is genuinely ambiguous between the two), so two
identically-trained models with different random initialization
disagreeing on borderline pigmented lesions is expected model behavior,
present in-domain as much as out, and orthogonal to whether the input
crop itself is informative. `crop_contrast` would never have caught
this, because these crops aren't low-quality — they're genuinely
ambiguous instances.

## Conclusion

The two signals **do not corroborate each other cleanly, and that is
the useful result, not a failure of either.** `crop_contrast` catches
"is there enough visual information in this crop to trust any
diagnosis" (fires hardest on ACK, moderately on BCC). Ensemble
disagreement catches "is this instance intrinsically ambiguous between
plausible diagnoses" (fires hardest on MEL, moderately on BCC — the
class flagged by both). They are complementary, not redundant: relying
on either alone would miss failure modes the other catches (ensemble
alone misses ACK's confidently-wrong-together pattern; crop_contrast
alone misses MEL's genuine boundary ambiguity). This is itself the
argument for CV-8 receiving CV-6 (and CV-5, CV-7) as convergent inputs
rather than a single collapsed score — exactly the architecture already
locked in `docs/project_state.md`.

## Decision

Per the spec's anti-rabbit-hole boundary: three signals, evaluated once
against the pre-committed corroboration question, documented honestly
including the partial/surprising result. No fourth signal added to force
a cleaner story. No abstention policy built — that is CV-8's job.
