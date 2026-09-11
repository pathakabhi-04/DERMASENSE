# Step 1 result — melanoma recall, measured on a set big enough to mean something

**Date:** 2026-09-12
**Plan:** `docs/cv_metrics_improvement_plan.md` Step 1
**Checkpoint:** `checkpoints/isic2019_resnet50_weighted_best.pt` (8-class, ISIC-trained)
**Split:** ISIC2019 test — 3,554 images, **666 melanomas**
**Domain:** in-domain (ISIC-trained, ISIC-evaluated). No taxonomy mapping
invented, no cross-domain confound. This is the model's *best case*.

## Pre-committed decision rule (fixed before running)

- MEL recall ≥ 0.80 → capability exists; the PAD-UFES transfer destroyed
  it. Fix the transfer strategy.
- 0.50–0.80 → partial; both data balance and transfer need work.
- < 0.50 → ISIC training itself is inadequate.

## Headline

**MEL recall = 0.5661** (377/666). Lands in the middle band, so *both*
training and transfer need work.

For context, the PAD-UFES figure that prompted this was 0.6667 — on
**9** melanomas. This measurement is on 666, so it is the first usable
estimate of melanoma sensitivity the project has.

## The number that actually matters

Headline recall understates the problem. What matters is not whether the
model says "MEL", but whether its prediction routes the user to a
clinician. Mapping the MEL confusion row through
`src/risk/action_mapping.py` (AK~ACK, NV~NEV, BKL~SEK):

| Predicted | n | Product action | Share of 666 |
|---|---:|---|---:|
| MEL 377 + BCC 30 + SCC 10 | 417 | `URGENT_EVALUATION` | 62.6% |
| AK 9 | 9 | `EVALUATE_SOON` | 1.4% |
| **NV 167 + BKL 69** | **236** | **`MONITOR` — "low risk"** | **35.4%** |
| DF 4, VASC 0 | 4 | no 6-class analogue | 0.6% |

**Sent to a clinician: 64.0%. Told "low risk": 35.4%.**

Misclassifying a melanoma as BCC or SCC is survivable — it still yields
`URGENT_EVALUATION`, and the patient sees someone. Misclassifying it as
NV or BKL is the failure that can kill: the user is told to monitor.

**167 melanomas called a benign nevus.** One in four.

## Why this is better news than it looks

It isolates where the loss happens. 62.6% of melanomas already route
correctly, so the pipeline is not blind to melanoma — it is *confidently
wrong in a specific direction*: MEL→NV. That is a decision-boundary
problem between two classes, which is far more tractable than "the model
cannot see melanoma".

It also means a large share of the gap is recoverable **without any new
data**, by changing what the model is optimised for. The current
objective treats MEL→NV and MEL→BCC as equally wrong. Clinically they
are not remotely equal.

## What this implies for Step 2

The plan said "retrain with melanoma represented". This result sharpens
it:

1. **Optimise the MEL→NV boundary specifically**, not Macro-F1. A
   cost-sensitive loss that penalises melanoma→benign far more than
   melanoma→other-malignant targets the 236, not the 289.
2. **Evaluate on "routed to a clinician", not class accuracy.** That is
   the product's real metric and it is what §4 of the plan asks for.
   Build it as a standing evaluation now.
3. **Consider a malignant/benign head** alongside the 6-class head. The
   binary question — does this need a doctor? — is easier than the
   6-way one and is what the product actually asks.

## Caveats, stated

- ISIC2019 is dermoscopy. Real phone photos are harder, so 0.5661 is an
  **upper bound** on deployed performance, not an estimate of it.
- The deployed checkpoint is the PAD-UFES-transferred 6-class model, not
  this one. Its melanoma behaviour on 666 images is still unmeasured —
  that is the obvious next run.
- The AK~ACK / NV~NEV / BKL~SEK correspondence is used only to map
  *predictions to actions*, not to relabel data. The 6-vs-8-class
  taxonomy question stays open.
