# Should we retrain? Yes — and here is the evidence that settles it

**Date:** 2026-09-12
**Script:** `scripts/evaluate_malignant_threshold.py`
**Sample:** ISIC2019 test — 666 melanomas, 2,051 benign (NV + BKL).
Deployed checkpoint, no retraining, same forward pass.

## The question

Routing uses `probabilities -> argmax -> class -> action`, which
discards the distribution. Could routing on summed malignant
probability, `P(MEL)+P(BCC)+P(SCC)`, fix the melanoma→benign failures
for free?

**Pre-committed rule:** adopt thresholding and defer retraining if some
threshold reaches ≥0.90 melanoma routing at ≤0.50 benign referral.

## Answer: no threshold qualifies

| threshold | melanoma routed | benign referred |
|---:|---:|---:|
| argmax (baseline) | 0.718 | 0.232 |
| 0.15 | 0.806 | 0.380 |
| 0.10 | 0.842 | 0.439 |
| 0.05 | 0.883 | 0.534 |

The most permissive threshold tested reaches 0.883 while referring 53.4%
of benign lesions — failing both halves of the rule. **Retraining is
required.**

## Why — the finding that matters

At a threshold of 0.05, **78 melanomas (11.7%) are still not routed**.
Those lesions receive **less than 5% total malignant probability**.

The model is not *uncertain* about them. It is *confidently wrong*.

No decision rule, threshold, or calibration can recover a lesion the
model assigns ~zero malignant probability. That is an information
problem in the weights, and only retraining addresses it. This is the
cleanest possible evidence that the remaining gap is not a
post-processing problem.

## What thresholding IS worth

Real, and available immediately:

> **threshold 0.15 → melanoma routing 0.718 → 0.806 (+0.089)**
> at benign referral 0.232 → 0.380 (+0.148)
>
> **59 more melanomas routed to a clinician, with no retraining.**

The cost is referring 38% of benign lesions instead of 23%. Whether
that is acceptable is a product decision about review capacity, not a
technical one — but note the asymmetry: a wrongly-referred benign lesion
costs one consultation, a missed melanoma can cost a life.

**Recommendation:** adopt the threshold as an interim while retraining
proceeds. It is strictly better on the only safety-critical axis, it
costs nothing, and it does not preclude anything.

## What retraining must fix

Not "accuracy". Specifically: **the 78 melanomas that receive near-zero
malignant probability.** That is the target, and it is now measurable —
any retraining run can be judged by whether that count falls, on the
same 666 images.

This also confirms the objective change from
`docs/cv_metrics_improvement_plan.md` Step 2: a loss that treats
melanoma→benign as far costlier than melanoma→other-malignant attacks
exactly this population. Macro-F1 does not.
