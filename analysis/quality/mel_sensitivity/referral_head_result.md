# Referral head, evaluated honestly — helps, but does not clear the bar

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
