# CV-8 contract delta — v1.2 (CV-4b referral head)

**Date:** 2026-09-12 · **Supersedes nothing in 1.1; purely additive.**
**Measurement:** `analysis/quality/mel_sensitivity/referral_head_result.md`

## What changed for a consumer

| | |
|---|---|
| `contract_version` | `"1.1"` → `"1.2"` — a MINOR bump; your parser already accepts higher minors |
| `quality_flags` | may now contain `REFERRAL_HEAD_RAISED_FROM_LOW` |
| `risk_category` | may be `MEDIUM` where 1.1 would have said `LOW` |
| `uncertainty.requires_review` | set `true` more often — 3 of the 5 delivered fixtures changed |
| everything else | unchanged: no key moved, no field changed shape |

**No parser change is required.** Re-pull `docs/cv8_sample_outputs/`.

## Why

The product question is "does this need a doctor?". Nothing was asking
it. Referral was inferred from which of six classes won the argmax, so a
melanoma that lost argmax to NEV was reported as `risk_category: LOW`.

CV-4b is a logistic head on the same frozen backbone features CV-4
already computes, trained refer-vs-benign. On the untouched ISIC2019
test split (666 melanomas), fit on train, threshold chosen on val:

| approach | melanoma routed | benign referred |
|---|---:|---:|
| argmax (what 1.1 shipped) | 0.7180 | 0.2320 |
| best 6-class probability threshold | 0.8060 | 0.3800 |
| **CV-4b referral head** | **0.9024** | 0.3940 |

**+120 melanomas of 666**, with no backbone retraining. The signal was in
the representation throughout; the 6-way softmax was discarding it.

## How it acts — a floor, not a ratchet

If the head says refer: `requires_review` becomes true, and
`risk_category` may not be `LOW`. `MEDIUM` and `HIGH` are untouched.

It is deliberately weaker than CV-7's one-step escalation. The head was
validated on refer-vs-benign only; it has no calibrated claim about
*degree* of risk, so promoting `MEDIUM` to `HIGH` would assert something
unmeasured. Preventing `LOW` is exactly what it was validated to do.

**It never touches `native_class`.** Narration still comes from CV-4.
Referral and diagnosis are now separate signals, because deriving one
from the other is what lost the melanomas.

## What you should know honestly

- **Dermoscopy only.** Every number above is ISIC2019. Phone photos are
  unmeasured and will be worse. Largest open unknown.
- **39.4% benign referral is a real cost**, against 23.2% before. That
  roughly doubles review volume — accepted deliberately as a staffing
  trade to clear the ≥0.90 rule, not a free win.
- **One split, one seed.** Confirm across seeds before relying on it.
- The safety gate already routed `MONITOR → REVIEW`, so a clinician saw
  these lesions before. What changes is that the **user** is no longer
  told `LOW` about a lesion the head flags.

## Rollback

Delete `checkpoints/referral_head/referral_head.json`, or pass
`referral_head_path=None`. CV-8 then behaves exactly as 1.1. There is a
test for this.
