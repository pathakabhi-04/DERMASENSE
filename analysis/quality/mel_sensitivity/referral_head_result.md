# Referral head, evaluated honestly

> **UPDATE (2026-09-12): refitted on the full train split, the rule is
> MET.** 0.9024 melanoma routing at 39.4% benign referral, threshold
> still chosen on val. Test AUC 0.8697 → **0.9050**. The val-only fit
> below (0.8498) was data-starved, not a ceiling. See §"Refit on train".

**Date:** 2026-09-12 · **Script:** `scripts/train_referral_head.py`
**Fit:** ISIC2019 **val** (3,304 usable). **Evaluate:** ISIC2019 **test**
(3,473 usable, 666 melanomas), never touched during fitting **or**
threshold selection.

## Result

| approach | melanoma routed | benign referred |
|---|---:|---:|
| argmax (shipped 6-class) | 0.7180 | 0.2320 |
| best probability threshold on 6-class | 0.8060 | 0.3800 |
| **referral head (held out)** | **0.8498** | 0.3974 |

Refer-vs-benign AUC on test: **0.8697**.

**The ≥0.90 target is not met.**

## Correction to the earlier probe

`binary_probe_result.md` reported 0.907 melanoma at 30% benign referral
and concluded the head "clears the ≥0.90 target". That measurement was
**out-of-fold on the test split itself** — a decodability probe, not a
generalisation estimate. The caveat was stated there; it turned out to
be the operative one.

Held out properly the number is **0.8498 at 39.7%**, and the operating
point chosen on val (30% benign) drifted to 39.7% on test — so the
threshold did not transfer cleanly either.

The qualitative conclusion survives: malignant/benign signal is present
in the frozen backbone and the 6-class softmax discards it. The
magnitude does not survive.

## Is it still worth shipping?

Compared at a like-for-like operating point (~38–40% benign referral):

- 6-class probability threshold: 0.806 melanoma @ 0.380 benign
- referral head: **0.850 melanoma @ 0.397 benign**

**+4.4 points of melanoma routing for +1.7 points of benign referral.**
Real, modest, and obtained with no backbone retraining — but not the
step-change the probe implied. 29 more melanomas out of 666.

## The most likely reason it underperforms, and the next test

The head was fitted on **3,304 samples in 2,048 dimensions**. That is a
thin regime; the earlier probe effectively saw more data through 5-fold
CV over a larger pool. ISIC **train** has 18,402 images — 5.5× more —
and is the obvious next fit.

**Pre-committed rule:** refit on ISIC train, evaluate on the same
untouched test. If melanoma routing at ≤40% benign referral reaches
≥0.90, the head ships. If it lands below 0.87, added data is not the
constraint and the ceiling is the frozen PAD-UFES representation —
which would make backbone retraining (on ISIC, for a referral
objective) the justified next spend rather than an avoidable one.

Cost: ~20 minutes of feature extraction from the external drive, no GPU.

## Standing caveats

- Still dermoscopy. Phone photos unmeasured and will be worse.
- The shipped gate already routes every `MONITOR` to `REVIEW`, so this
  head changes *how many* lesions reach the review queue, not whether
  melanomas can be auto-released as benign.
- DF/VASC excluded rather than mapped; the taxonomy question stays open.


---

# Refit on train — the rule is met

**Fit:** ISIC2019 **train** (18,402 images, 5.5× the val-only fit).
**Threshold:** chosen on **val**. **Reported:** untouched **test**.

Test AUC **0.9050** (val-only fit: 0.8697).

| val budget | test benign referred | test melanoma routed |
|---:|---:|---:|
| 25% | 0.2452 | 0.7898 |
| 30% | 0.2882 | 0.8288 |
| 35% | 0.3364 | 0.8664 |
| **40%** | **0.3940** | **0.9024** |
| 45% | 0.4544 | 0.9204 |

**Pre-committed rule:** ≥0.90 melanoma at ≤40% benign referral → ship.
**Met: 0.9024 at 0.3940.**

## Against the shipped decision rule, same test set

| approach | melanoma routed | benign referred |
|---|---:|---:|
| argmax (shipped 6-class) | 0.7180 | 0.2320 |
| best probability threshold on 6-class | 0.8060 | 0.3800 |
| **referral head (train-fitted)** | **0.9024** | 0.3940 |

**+18.4 points of melanoma routing over the shipped rule**, for +16.2
points of benign referral. In absolute terms on this test set: **120 more
melanomas routed to a clinician**, out of 666.

## What this settles

1. **No backbone retraining is indicated.** A logistic head on frozen
   PAD-UFES features clears the target. The GPU spend the plan
   contemplated is not needed for this.
2. **The val-only result was data starvation, not a ceiling.** 3,304
   samples in 2,048 dimensions underfit; 18,402 does not. Worth
   remembering before concluding "the representation can't do it".
3. **Every earlier negative was tuning the wrong objective.** Class
   weighting, capacity and SupCon all optimised the 6-way softmax. The
   information was in the backbone throughout.

## What it does not settle

- **Still dermoscopy.** Phone photos are unmeasured and will be worse.
  This is the largest remaining unknown and no amount of ISIC work
  closes it.
- **39.4% benign referral is a real cost.** The shipped `MONITOR →
  REVIEW` gate reviews 18.8% today, so this roughly doubles queue
  volume. That is a staffing decision, not a technical one.
- **One split, one seed, one architecture.** Before shipping: confirm
  across seeds, and confirm the head composes with CV-8 rather than
  fighting it.
- The head answers "does this need a clinician?" only. The 6-class
  prediction must still drive narration — this is a routing signal, not
  a diagnosis.
