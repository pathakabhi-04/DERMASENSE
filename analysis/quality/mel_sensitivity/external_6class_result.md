# External 6-class evaluation: melanoma recall is finally measurable

**Date:** 2026-09-14
**Set:** `external_6class_eval_set.csv` — DDI + DDI-2 + Fitzpatrick17k
mapped onto the PAD label space (792 images, **107 melanomas**), mapping
pre-committed in `scripts/build_external_6class_eval.py`
**Run:** `python -m scripts.evaluate_pipeline_end_to_end --split external`
**Raw:** `analysis/product_eval/cv1_cv4_assembly/external_{predictions.csv,summary.txt}`

This closes Step 1 of `docs/cv_metrics_improvement_plan.md` ("measure
melanoma recall on a set big enough to mean something") for the 6-class
task, and builds the Step 4 standing end-to-end evaluation.

## The headline, stated unconditionally over all 107 melanomas

| outcome | n | share |
|---|---:|---:|
| reaches a clinician | 62 | **57.9%** |
| assessed → `MONITOR` | 37 | **34.6%** ← the dangerous failure |
| never assessed (no candidate / quality) | 8 | 7.5% |

Conditional on being assessed: **62/99 = 0.6263, 95% CI [0.53, 0.72]**.

For reference, PAD-UFES test gave 6/9 = 0.6667 with a CI of roughly
[0.30, 0.93]. The point estimates are similar; the difference is that
this one is *measured*. The CI is four times tighter.

**"Reaches a clinician" is the right criterion, not "classified as
MEL"** (plan Step 4): a melanoma called BCC still yields
`URGENT_EVALUATION` and is a good outcome; called NEV it yields
`MONITOR`, which is the failure that matters.

## Per-class, on the 676 assessed images

| class | n | recall | CI | routed to clinician |
|---|---:|---:|---|---:|
| ACK | 27 | 0.1852 | [0.08, 0.37] | 0.5556 |
| BCC | 134 | 0.4179 | [0.34, 0.50] | 0.8134 |
| **MEL** | **99** | **0.1818** | **[0.12, 0.27]** | **0.6263** |
| NEV | 195 | 0.6256 | [0.56, 0.69] | 0.2667 |
| SCC | 99 | 0.2626 | [0.19, 0.36] | 0.7980 |
| SEK | 122 | 0.1967 | [0.14, 0.28] | 0.3279 |

**Macro recall 0.3117**, against the in-domain PAD-UFES Macro-F1
baseline of 0.5996. Roughly half the in-domain figure — consistent with
the domain shift already measured in `domain_shift_second_source.md`,
and a reminder that this number confounds "CV-4 is weak" with "these are
three sources CV-4 never trained on". It is not a clean CV-4 accuracy
claim, and should not be quoted as one.

Discrimination is weak but real: melanoma routes at 0.63 against nevi at
0.27. Melanomas are predicted as
`BCC 31, NEV 30, MEL 18, SCC 12, SEK 7, ACK 1` — the model far more
often says *some malignancy* than *melanoma*, which is why the routing
number (0.63) is three times the classification number (0.18) and why
routing is the metric that describes the product.

## A methodological trap this run walked into, and out of

The first run of this evaluation disabled CV-2, on the assumption that
external clinical close-ups are pre-framed like PAD-UFES. It reported
**melanoma routing of 0.7250**.

That number was wrong, and wrong in the flattering direction. CV-1.5
routes **357 of 786 (45%)** of these images to `wide_field`, and with no
detector loaded every one of those became a silent `NO_CANDIDATES` —
so the evaluation scored only the 429 images the router found easy.
Re-running with CV-2 enabled pulled melanoma routing down to **0.6263**,
a **−0.10** correction, and grew the assessed set from 429 to 676.

Recorded because the failure mode is general: a stage that silently
drops inputs turns an evaluation into a measurement of the subset that
stage found easy. The drop rate here (46%) was large enough to notice;
a 5% drop with the same bias would not have been.

## This refutes a correction made earlier the same day

`product_scope_and_readiness.md` was edited today to say that CV-1.5
routes pre-framed photos past CV-2, so *"for a product where the user
photographs one lesion close-up, CV-2 is not in the path and its recall
does not multiply into melanoma sensitivity at all."*

**That was too strong, and this run is the evidence against it.** It
holds for PAD-UFES, which is pre-framed *by construction*. It does not
hold for real clinical photographs from other sources:

| source | pre_framed | wide_field |
|---|---:|---:|
| DDI | 180 | 131 (42%) |
| DDI-2 | 56 | 175 (76%) |
| Fitzpatrick17k | 193 | 51 (21%) |

So CV-2 **is** in the path for roughly half of real-world close-ups, and
its recall does partly gate them. On the images routed to it here, CV-2
returned no candidate for **110 of 357 (30.8%)** — worse than the 19%
it records on iToBoS.

Whether DDI-2's 76% is router error or genuinely wider framing is not
settled here: the composition audit measured DDI-2's median lesion area
fraction at 0.249 against DDI's 0.360, so some of those images may
legitimately be wider shots. Either way the consequence for the product
is the same.

> **Settled 2026-09-14 — it is genuine framing, not router error**
> (`cv1_5_routing_resolution.md`). Images the router calls `wide_field`
> have a median whole-frame lesion area fraction of 0.236 against
> `pre_framed`'s 0.402, separated at every quantile, with PAD-UFES's
> true-pre-framed 0.312 sitting between them. The router is splitting a
> real continuum near the right place. Forcing everything pre-framed
> moves melanoma reach-rate by +1.9 points — inside the noise — so
> routing is not the lever either. Any reading of this section as
> "the router is misrouting" is superseded.

This also exercises the **proxy-label caveat the router spec itself
flagged**: CV-1.5 scored 1.000/1.000 on a held-out set where dataset
identity (PAD-UFES vs iToBoS) stood in for verified per-image framing.
A third source is the first real test of that, and 45% wide_field is
what it produced.

## What this changes

1. **CV-4's melanoma recall is now measurable**, and it is 0.18
   classified / 0.63 routed under domain shift. Any future change can be
   scored against this instead of against 9 images.
2. **The standing end-to-end evaluation exists** (plan Step 4) and runs
   with one command on three independent external sources.
3. **CV-2 is back on the critical path** for real-world input, which
   re-opens the tiling/SAHI question that `cv2_status.md` deferred — its
   re-entry gate was "if CV-2's miss rate turns out to be a dominant
   pipeline failure", and 30.8% no-candidate on real clinical photos is
   the first evidence that bears on it.
4. **The dangerous number is 34.6%** — melanomas that were assessed and
   sent to `MONITOR`. The shipped safety gate routes every `MONITOR` to
   human review, so this is a review-queue load, not a reassurance
   delivered to a patient. It becomes a patient-safety number the moment
   `MONITOR` is auto-released, which §2.4 of the readiness doc already
   forbids.

## Caveats

- The label mapping is a judgment call in places; the calls are listed
  in `build_external_6class_eval.py`'s docstring. Bowen's disease is
  excluded from SCC by default (`--include-scc-in-situ` runs the variant).
- ACK (n=27) is thin; its recall carries a wide interval.
- These are three sources the model never trained on, so every number
  here is CV-4-under-shift, not CV-4 in-domain.
- 1,108 of 1,900 source images had no PAD analogue and were excluded
  rather than force-mapped.
