# Multi-domain backbone training: scope

**Date:** 2026-09-14
**Status:** design, pre-committed. Nothing here is chosen after seeing a
result.
**Supersedes nothing** — this is a *new hypothesis*, not a retry of
`cv4b_backbone_finetune_design.md`. Read that and
`cv4b_backbone_finetune_result.md` first.

## 1. Why this is not just "run it again"

The fine-tune's pre-committed §12 said no hyperparameter iteration and no
escalation on momentum. This clears that bar for two independent reasons,
and both should be stated plainly rather than assumed:

**(a) The previous experiment had a defect.** PAD-UFES was never in the
training mix. The bundle contains `isic2019`, `ddi`, `ddi2`,
`fitzpatrick17k` and nothing else — 1,610 clinical training images from a
distinct clinical domain were simply omitted. That is a bug in the
experiment, not a result, and it justifies a re-run on its own.

**(b) The LOSO folds could not have tested what they were named for.**

| fold | clinical domains in training |
|---|---:|
| `loso_atlas` | Stanford only — **1** |
| `loso_stanford` | atlas only — **1** |

**Domain-invariant features cannot be learned from a single domain**;
there is no cross-domain variation to be invariant to. So the LOSO folds
measured *single-source transfer*, which is a strictly harder and much
less informative problem than domain generalization. The 0.63–0.67 result
is real, but it does not license the conclusion "multi-domain training
would not help", because multi-domain training is not what ran.

## 2. Hypothesis

> A shared backbone trained across **three or more** distinct domains,
> with per-domain heads absorbing each domain's own label prevalence and
> calibration, learns features that transfer to a **held-out** domain
> better than a single pooled head trained on one or two.

## 3. What changes, concretely

**Architecture — already half-built.** `DermaSenseNativeClassifier`
already carries `pad_ufes_head` (6-class) and `isic2019_head` (8-class)
on a shared ResNet-50. This adds:

- one **binary head per training domain** (isic, pad, stanford, atlas) —
  auxiliary tasks whose only job is to shape the backbone;
- one **pooled domain-agnostic binary head** trained on all domains at
  once. **This is the head that ships**, because a user's phone is an
  unseen domain with no head of its own. Per-domain heads that cannot be
  selected at deployment are a representation-learning device, not a
  product.

**Trainable:** layer4 + all heads. Layers 0–3 frozen, asserted by exact
parameter count, as before.

**Optional second arm — domain-invariance penalty.** A gradient-reversal
domain classifier (DANN-style) on the shared features, penalising the
backbone for encoding *which source* an image came from. This targets the
measured failure directly. Run **only if arm 1 clears its bar**; two
architectural changes at once cannot be attributed.

## 4. Domains

| domain group | source | train images | character |
|---|---|---:|---|
| `isic` | ISIC2019 | 18,062 | dermoscopy |
| `pad` | PAD-UFES | 1,610 | clinical, Brazil — **newly added** |
| `stanford` | DDI + DDI-2 | ~990 | clinical, biopsy-proven |
| `atlas` | Fitzpatrick17k | ~440 | teaching atlas |

DDI and DDI-2 stay one group: both Stanford AIMI, and DDI has no patient
IDs to rule out overlap (`build_cv4b_splits.py`).

PAD-UFES maps to the binary task with no new taxonomy decisions — it is
the label space the referral head's `REFER`/`BENIGN` convention was
written against (1,265 refer / 345 benign in train).

## 5. Folds

Leave-one-clinical-domain-out, with **three** training domains each time:

| fold | trains on | tests on | test melanomas |
|---|---|---|---:|
| `dg_atlas` | isic + pad + stanford | Fitzpatrick17k | 83 |
| `dg_stanford` | isic + pad + atlas | DDI + DDI-2 | 25 |
| `dg_pad` | isic + stanford + atlas | PAD-UFES test | 9 |

`dg_pad` is reported but **not gated** — 9 melanomas is the measurement
problem that started all of this. It is included because PAD-UFES is a
genuinely held-out clinical domain and its *malignant-vs-benign* AUC
(n=352) is usable even though its melanoma count is not.

Patient grouping and the no-leak assertions carry over unchanged.

## 6. Pre-committed decision rule

Compared against the **same folds' predecessors** (`loso_atlas` 0.6682,
`loso_stanford` 0.6257):

| criterion | bar |
|---|---|
| held-out-domain AUC, `dg_atlas` and `dg_stanford` | **≥ 0.75 both** |
| ISIC test AUC | ≥ 0.85 (unchanged; no forgetting) |
| held-out benign referral | ≤ 0.45 |

**≥ 0.75 is the falsifier.** It sits above the +0.02 that pooling noise
could produce and below the 0.80 the previous design asked for, because
the claim here is "domain count is a mechanism", not "this ships".

Outcomes, decided now:

- **Both folds ≥ 0.75** → domain count is a real mechanism. Proceed to
  the DANN arm, and re-scope toward deployment.
- **Only the pooled/seen numbers improve again** → the constraint is
  per-source data, as three investigations have already concluded.
  **Stop. This is the last domain-generalization attempt.**
- **ISIC floor breached** → mixing ratio wrong; one corrected retry, as
  before.

Honest prior: the domain-generalization literature frequently finds that
DG methods fail to beat plain pooled training. A null result here is a
genuinely likely outcome and would be worth recording as one.

## 7. Data logistics — no re-upload

**ISIC and everything already uploaded stays put.** Verified against the
live volume: 27,322 objects / 2.72 GB, including all 26,739 bundle
images.

Only PAD-UFES is new:

| item | measured |
|---|---:|
| PAD-UFES images (train+val+test) | 2,298 |
| mean size at 320px q100 | 71 KB (20-image sample) |
| **incremental upload** | **~159 MB** |
| bundle after | ~1.86 GB |

`aws s3 sync` transfers only what is missing, so rebuilding the bundle
locally and re-syncing uploads the ~159 MB of new PAD files and skips the
26,739 that already match. **No volume expansion is needed** — the delta
is under 4% of what is already there.

One cleanup worth doing: `.cache/pip` is on the volume (54 objects),
which means pip installed into `/workspace` rather than the image, as §12
warned it might. Harmless, but it is volume-billed space.

## 8. Cost

Same shape as the last run: layer4-only, ~21K images/epoch after adding
PAD, ≤30 epochs, three folds. **~2–3 GPU-hours, ~$3–10.** The pre-flight
gate and the bundle/verify path are already built and unchanged.

## 9. What this refuses to do

- No full-backbone unfreeze.
- No hyperparameter search; one configuration per arm.
- No DANN arm unless arm 1 clears its bar.
- No re-running a fold because its number disappointed.
- Test touched once per fold, by `evaluate_cv4b_finetune.py`.
