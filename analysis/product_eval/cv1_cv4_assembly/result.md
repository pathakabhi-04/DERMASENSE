# CV-1 → CV-4 Pipeline Assembly — Result

**Status:** Assembled and measured. Regression check PASSES cleanly
(100% agreement with CV-4 standalone). Two new product-level findings
surfaced that pairwise component validation could not have shown.

Spec: `docs/cv1_cv4_assembly_spec.md`. Code:
`src/inference/orchestrator.py`. Eval:
`scripts/evaluate_pipeline_end_to_end.py`.

## Pre-framed branch — PAD-UFES test, 352 images

Regression check against CV-4 standalone, computed on **exactly the
images the assembly scored** (comparing against the whole-set baseline
number would be survivorship-biased, since the assembly can drop images
at CV-1):

| | assembly | CV-4 alone |
|---|---|---|
| Macro-F1 | 0.6351 | 0.6351 |
| Tier-1 errors | 31 | 31 |
| **Per-image agreement** | **100.0%** | |

The assembly reproduces CV-4's standalone predictions exactly on all 303
images it assessed. The preprocessing-mismatch risk flagged in the spec
(cv2 vs PIL resize paths) did **not** materialize — routing a full-frame
box through `crop_and_normalize` is genuinely equivalent to feeding the
whole image.

**Finding 1 — CV-1 rejects 13.6% of real clinical images.**
49/352 were never assessed (48 QUALITY_REJECTED, 1 routed `wide_field`).
This is the known "CV-1 validated on synthetic degradation only"
caveat, now measured on real data: roughly 1 in 7 real clinical photos
gets a "retake" rather than an assessment. 19 of the dropped lesions are
high-risk (17 BCC, 2 MEL).

Mitigating detail: the drops are **not** biased toward high-risk —
38.8% of dropped images are high-risk against a 50.9% base rate, so
CV-1 is not preferentially discarding dangerous lesions. But the
absolute count matters for the product, and this is precisely the
population the capture-guidance layer exists to serve.

## Wide-field branch — iToBoS test, 1000-image seeded random sample

**No accuracy claim is made** — iToBoS carries no diagnosis labels.
Structural propagation only.

| Outcome | Count | Share |
|---|---|---|
| ASSESSED | 655 | 65.5% |
| QUALITY_REJECTED (CV-1) | 215 | 21.5% |
| NO_CANDIDATES (CV-2) | 130 | 13.0% |

**Finding 2 — 34.5% of wide-field submissions produce no assessment at
all.** CV-1 drops 21.5%, then CV-2 finds nothing in a further 13.0%.
Neither rate is alarming alone; stacked, they are. This compounding is
exactly what pairwise component validation could not reveal, and is the
main reason this assembly was worth doing before building CV-5/6/7.

The 13.0% is **not** comparable to CV-2's documented ~19% miss rate:
that figure was measured on lesion-containing images only, whereas this
denominator is all sampled images (iToBoS deliberately includes
zero-lesion images) and is measured *after* CV-1 already removed 21.5%.
Different denominators — not evidence CV-2's behaviour changed.

**Finding 3 — alarm fatigue on the wide-field branch.**
Among assessed images: 32.7% escalate to URGENT_EVALUATION and 85.0%
require human review.

Mechanism, from candidate-level data: 3,082 candidates across 655 images
(mean 4.71, median 3, max 27), with a **12.9% per-candidate high-risk
rate** (357 BCC, 39 MEL, 1 SCC). Under most-severe aggregation a single
high-risk prediction anywhere escalates the whole image, so 12.9%
per-candidate compounds into 32.7% per-image.

That per-candidate rate is almost certainly inflated: CV-4 is
classifying TBP crops, a domain it was never validated on (it is
validated on PAD-UFES clinical close-ups). 357 BCC predictions in a
routine screening population is not clinically plausible.

**This is not an argument to weaken the aggregation rule**, which is
correctly conservative — a screening tool must not hide a high-risk
candidate behind an averaged image-level verdict. It is evidence that
the wide-field branch's *classification* step is out of domain, which is
the already-tracked CV-4 domain gap, now quantified end-to-end.

## What worked as designed

- **Outcome typing held.** The `UNKNOWN` action count (345) exactly
  equals QUALITY_REJECTED + NO_CANDIDATES. Not one unassessed image
  leaked into a real risk category — the silent-miss concern is
  structurally contained.
- **CV-1.5 was consistent.** All 785 iToBoS images reaching routing went
  `wide_field`; all 303 assessed PAD-UFES images went `pre_framed`
  (1 exception out of 352). Real-world behaviour matches the 150+150
  held-out result.
- **CV-3 degeneracy was low here** — 1.0% on wide-field candidates,
  far better than the 22% measured on CV-2 detection crops in the CV-3
  domain audit. Worth noting the two are measured on different crop
  populations.

## Decision

The assembly is sound and does not degrade CV-4. Per the spec's
anti-rabbit-hole boundary, no tuning was attempted and none is proposed
here. The three findings above are recorded as measured limitations, and
the two that matter for prioritisation (CV-1's real-image rejection
rate; CV-4's out-of-domain behaviour on TBP crops) are better targets
for future work than any of CV-5/6/7 would have been chosen blind.
