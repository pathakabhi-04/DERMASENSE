# CV-1.5 routing: not the lever, and the router is not wrong

**Date:** 2026-09-14
**Script:** `scripts/resolve_cv1_5_routing.py` (design and decision rule
pre-committed in its docstring, before the run)
**Raised by:** `external_6class_result.md` — the router sends 45% of real
clinical close-ups to `wide_field`, where CV-2 then finds nothing 30.8%
of the time.

## The question, posed answerably

There are no ground-truth framing labels for DDI/DDI-2/Fitzpatrick, so
"is the router correct?" cannot be answered directly. The operational
form can be: **does forcing every image down the pre-framed branch route
more melanomas to a clinician than letting the router decide?**

| arm | melanomas reaching a clinician (of 107) | benign referral |
|---|---:|---:|
| A — router decides | 62 = 0.5794 [0.48, 0.67] | 0.2902 |
| B — forced `pre_framed` | 64 = 0.5981 [0.50, 0.69] | 0.3484 |
| **delta** | **+1.9 points** | **+5.8 points** |

Pre-committed rule: `|delta| < 5 points` → **routing is not the lever.**
That is the branch this landed in, and the verdict stands as written
before the run.

Bypassing the router entirely recovers **all 110** `NO_CANDIDATES`
images — arm B assesses 786 of 792, against arm A's 676 — and converts
**two** additional melanomas, while referring 5.8 points more benign
lesions. That is not a trade worth making, and more importantly it means
neither CV-1.5 nor CV-2 is what is costing melanoma sensitivity here.

## The router is tracking something real

Whole-frame lesion area fraction, measured by CV-3 on every image in arm
B, split by what the router had said:

| router said | n | p10 | p25 | median | p75 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| `pre_framed` | 429 | 0.075 | 0.196 | **0.402** | 0.546 | 0.701 |
| `wide_field` | 357 | 0.026 | 0.093 | **0.236** | 0.356 | 0.488 |

PAD-UFES, true pre-framed by construction, has a median of 0.312 —
between the two. The images CV-1.5 calls `wide_field` genuinely do have
a smaller lesion-to-frame ratio, by a wide margin at every quantile.

**So the 45% is not a router failure.** These three sources contain a
continuum of framing, and the router is splitting it about where
PAD-UFES sits. Calling it "misrouting" — as
`external_6class_result.md` tentatively did — was wrong, and this
supersedes that reading. The distributions overlap heavily, so
individual decisions near the boundary are arbitrary, but the split is
not arbitrary.

## The finding that actually matters: CV-2 misses benign lesions

Of the 110 images CV-2 returned no candidate for, the true classes were:

```
NEV 66, SEK 16, BCC 11, MEL 8, SCC 6, ACK 3
```

**82 of 110 (75%) are benign.** Only 8 are melanomas. And of those 8,
once forced through the pre-framed branch and actually classified, only
**2 reached a clinician** — the other 6 were predicted `NEV`.

This is the second-order result, and it is more useful than the first:

> CV-2's 30.8% no-candidate rate on real clinical photos costs almost
> no melanoma sensitivity, because what it drops is overwhelmingly
> benign — and because the melanomas it does drop are mostly ones CV-4
> would have misclassified anyway.

That bears directly on `cv2_status.md`'s deferred tiling/SAHI
refinement, whose re-entry gate was *"IF that evaluation shows CV-2 miss
rate is a dominant contributor to end-to-end failure"*. **It is not.**
On this evidence the tiling work stays deferred, and the gate is now
answered with data rather than left unsatisfiable.

## Where the melanoma sensitivity actually goes

Tracing all 107 melanomas through arm B (every image assessed, no
detection or routing losses at all):

- 64 reach a clinician
- 43 are assessed and sent to `MONITOR`

With detection and routing removed as factors entirely, **40% of
melanomas still fail to reach a clinician.** The loss is in CV-4's
discrimination on clinical photographs it never trained on — the same
zero-shot-transfer failure the backbone fine-tune could not fix
(`cv4b_backbone_finetune_result.md`).

Three independent investigations now point at the same constraint:
per-source labelled data, not architecture, detection, or routing.

## What this closes

1. **CV-1.5 needs no change.** It is not misrouting; it is splitting a
   real continuum near the right place. No retraining with a third
   source is justified on this evidence.
2. **CV-2 tiling stays deferred**, now for a measured reason rather than
   an unsatisfiable gate.
3. **Neither is the lever.** The next real lever is CV-4 under domain
   shift, and the only thing shown to move that is labelled data from
   the deployment source.

## Caveats

- 107 melanomas; the arm A/B difference (+1.9 points) is two images and
  well inside the overlapping confidence intervals. The conclusion is
  "no material difference", which is what the pre-committed rule asked,
  not "B is better".
- Benign referral is measured only on NEV+SEK, the two classes the
  product treats as not-requiring-review.
- Arm B disables CV-2 entirely; it is a counterfactual for attribution,
  not a proposed configuration.
- A methodological note: the first attempt at this run completed all 792
  images and then threw away the result on an over-strict assertion (it
  required every row to be `pre_framed`, but `QUALITY_REJECTED` returns
  before routing runs, so 6 rows legitimately had `framing=None`). The
  script now persists results before validating them.
