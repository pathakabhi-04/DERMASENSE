# ABCD from the mask: the hypothesis is falsified, and inverted

**Date:** 2026-09-15
**Script:** `scripts/evaluate_abcd_features.py` (scope corrections and
decision rule pre-committed in its docstring)
**Data:** 3,007 images with a usable CV-3 mask, across three source
groups — `stanford` (DDI + DDI-2), `atlas` (Fitzpatrick17k),
`pad` (PAD-UFES). 83 skipped for an unreadable image or a degenerate mask.

## The hypothesis

Four investigations showed the backbone's learned features collapse on
unseen capture sources, with lighting, white balance and compression as
the stated mechanism — all of which change **appearance**. Shape does
not. So: **geometry might transfer where texture did not**, giving the
narrow product a first-visit feature it currently lacks.

## Result: it does not, and the reason is more interesting than the verdict

Held out one source group at a time, melanoma vs benign (NEV+SEK):

| held out | AUC | melanoma / benign | CNN referral head |
|---|---:|---:|---:|
| stanford | 0.5963 | 24 / 399 | 0.60 |
| atlas | **unmeasurable** | 83 / **1** | 0.6924 |
| pad | 0.6073 | 51 / 468 | — |

Both measurable folds land at ~0.60 — **no better than the CNN they were
supposed to beat**, and far below the 0.70 bar. Bar not cleared;
`DOES NOT CLEAR` as pre-committed.

### The one number that appeared to pass was an artifact

The first run reported **atlas AUC 0.8675**, the single figure clearing
the bar. Fitzpatrick17k's mapped set contains **83 melanomas and one
benign lesion** — that AUC is computed against a single negative and
means nothing. A `MIN_PER_CLASS = 10` guard now returns `nan` there
instead, so the number cannot be misread later.

Worth stating plainly: the only result that looked like success was the
one with n=1 in a class. That is the second time in this series a
flattering number came from a degenerate comparison, after `dg_pad`.

## A and B carry no signal — and point the wrong way

Pooled single-feature AUCs, melanoma vs benign:

| feature | AUC | melanoma (median) | benign (median) |
|---|---:|---:|---:|
| asymmetry | **0.4449** | 0.2942 | 0.3411 |
| border irregularity (solidity) | **0.4750** | 0.1710 | 0.1923 |
| border irregularity (circularity) | **0.5023** | 0.7975 | 0.8013 |
| **colour variation** | **0.7008** | 15.40 | 11.63 |

The geometric features are not merely weak — **they are inverted**. By
these measures melanomas are *less* asymmetric and *less*
border-irregular than benign lesions, which is the opposite of the
clinical rule they were modelled on. Circularity is pure chance (0.5023).

The features do vary (asymmetry spans 0.08–0.72 at the 5th/95th
percentile), so this is not a constant-output bug. They vary, and what
they track is not melanoma.

**Most likely explanation, untested:** CV-3 is a segmentation model
trained to produce clean lesion outlines. A U-Net's masks are smooth
and regularised — it does not trace the fine border structure a
dermatoscopist reads, and it may produce *more* irregular outlines on
small or edge-touching lesions, which in these datasets skew benign.
The mask is a good segmentation and a poor border tracing, and those
are not the same artifact.

## The inversion

The hypothesis was: *geometry transfers, appearance does not.*

The result is the exact reverse. The only feature carrying melanoma
signal is **colour variation** (0.7008 pooled) — the appearance property
predicted to be most vulnerable to the domain shift that broke
everything else. And even it does not survive: the four-feature model
containing it reaches only ~0.60 on held-out sources.

That is worth recording precisely because the reasoning felt sound. A
plausible mechanism, argued in advance, was wrong in its direction.

## Decision

- **No ABCD feature ships**, in any direction. They do not clear the
  bar, and the geometric ones would be actively misleading if surfaced —
  a lesion flagged "irregular border" by a measure that runs backwards
  on melanoma is worse than saying nothing.
- **The first-visit gap stands.** Plan A's first visit still offers a
  measurement, a stored baseline, capture feedback, a clinician handoff,
  and general cited education — and no lesion-specific signal.
- **Colour variation is not pursued further.** At 0.70 pooled and ~0.60
  transferred it is the same story as every learned feature in this
  series, and chasing it would be the fifth attempt at a constraint
  already isolated four times.

## Caveats

- 24 and 51 melanomas in the two measurable folds; both AUCs carry wide
  intervals. The conclusion is "no better than the CNN", which those
  samples support, rather than a precise value.
- One implementation of each feature. Better asymmetry or border
  descriptors exist in the dermoscopy literature and were not tried —
  but the inverted direction suggests the mask itself is the limit, not
  the descriptor.
- Measured on CV-3's masks, not on expert lesion tracings. A conclusion
  about ABCD-from-our-masks, not about ABCDE as clinical practice.
