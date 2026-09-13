# CV-4b retrain/refit: scope, pre-committed before running

**Date:** 2026-09-13
**Status:** scoping — this is the plan, written before any refit is run,
per the same bounded-experiment discipline as the rest of this
investigation (question, sample, decision rule, fixed in advance).

## Why this, and why now

`domain_shift_second_source.md` settled that CV-4b's (the referral
head's) transfer failure is general across all three independent
sources tested (AUC 0.90 in-domain → 0.59–0.69 everywhere else), and
concluded refitting it on a broader-than-ISIC-only feature distribution
is justified independent of the still-open native-classifier question.
This is that refit, scoped concretely.

**What this is not**: not backbone retraining, not a GPU job. The head
is a logistic regression on frozen 2048-d ResNet-50 features
(`scripts/train_referral_head.py`) — this stays a linear probe. Only the
*data it's fit on* changes. That keeps this a cheap, CPU-only
experiment consistent with the current no-GPU-budget constraint, and it
isolates one question from a bigger one: does more diverse *training
data* fix transfer, before spending anything on unfreezing the
backbone.

## Data

| source | n (usable) | malignant | melanoma |
|---|---:|---:|---:|
| DDI | 656 | 171 | 21 |
| DDI-2 | 665 | 71 | 4 |
| Fitzpatrick17k (atlas) | 583 | 283 | 83 |
| **non-ISIC total** | **1904** | **525** | **108** |

ISIC train/val/test features are already cached
(`analysis/scc_bcc/isic2019_{train,val}_backbone_features.npz`,
`isic2019_test_scc_bcc_features.npz`) from the original fit. Non-ISIC
features do not exist yet and must be extracted once through the same
frozen backbone (`load_backbone_for_features()` in
`scripts/evaluate_domain_shift.py`, already verified against
`NativePredictor`'s exact construction).

## Label harmonization (a known approximation, not a new inconsistency)

The shipped head's label is *referral need*, not strict malignancy:
`REFER = (MEL, BCC, SCC, AK)` vs `BENIGN = (NV, BKL)` on ISIC — AK
(premalignant) is bucketed as refer because it needs a clinician, not
because it's malignant. DDI/DDI-2/Fitzpatrick only carry a
malignant/benign boolean, not a referral-need flag, so any AK-equivalent
premalignant case in those sources would fall into their "benign" bucket
even though the shipped head's own convention would refer it. This is
the same approximation `evaluate_domain_shift.py` already made silently
when computing cross-source AUC (`is_malignant` as the unified label) —
carried forward here for train/eval consistency, not introduced fresh.
Flagging it rather than hiding it: this is a source of mild label noise
in the non-ISIC benign pool, not a corrupted experiment.

ISIC keeps its existing `REFER`/`BENIGN` convention unchanged. Non-ISIC
sources use `is_malignant` as the label.

## Split discipline

Non-ISIC pool (1904 images) split **once**, stratified by
`(source, is_malignant)` so every source×label cell is represented
proportionally in each split — not stratified on melanoma specifically,
since melanoma is a reporting slice of the label, not the label itself.
Split 60/train, 20/val, 20/test, fixed seed, assignment written to a
CSV before any fitting happens so it's inspectable and cannot be
silently redone if a result disappoints.

Mirrors the existing ISIC discipline exactly:
- **fit**: ISIC train ∪ non-ISIC train
- **threshold**: chosen on ISIC val ∪ non-ISIC val (as now), never on test
- **eval**: ISIC test (untouched, same 1,148-ish rows as today) **and**
  non-ISIC test (held out, never seen during fit or thresholding),
  reported both separately per source and combined

**DDI-2's 4 total melanomas mean its test slice will carry approximately
0–1 melanoma.** That is not a flaw in the split, it's a fact about the
data — reported with a Wilson interval like every other small-n figure
in this project, not smoothed over. The *malignant* (not melanoma-only)
counts per split are large enough to be informative (~525 non-ISIC
malignant total, ~105 in a 20% test slice) — melanoma-routing on the
combined non-ISIC test set is the interpretable number; per-source
melanoma counts are context, not a gating metric on their own for
DDI-2 specifically.

## Pre-committed decision rule

**Primary question:** does adding non-ISIC training data recover the
held-out non-ISIC test AUC, without collapsing the ISIC test AUC?

- **Success:** non-ISIC held-out test AUC ≥ 0.80 (vs. today's 0.59–0.69
  uniform collapse) **and** ISIC test AUC stays ≥ 0.85 (vs. today's
  0.9050 — some regression from spending capacity on a broader
  distribution is expected and acceptable, a collapse is not).
- **Reported, not gated:** melanoma-routing rate on the combined
  non-ISIC held-out test, with a Wilson CI, at whatever benign-referral
  budget the val-chosen threshold implies (same 0.40-budget convention
  as today, re-derived on the combined val set).
- **If the bar is not met:** that is evidence the failure is
  representational — the frozen ISIC-only backbone's features don't
  carry what a linear probe needs, regardless of what data the probe is
  fit on — which argues for a different, bigger next step (backbone
  fine-tuning, which needs GPU budget not currently available) rather
  than iterating on this head's hyperparameters to chase the number.
  This experiment stops at that conclusion; it does not become a
  hyperparameter search.

## What happens next, concretely

1. Extract and cache non-ISIC features (`scripts/extract_domain_shift_features.py`, new).
2. Fixed stratified split, written to CSV (`analysis/quality/mel_sensitivity/cv4b_retrain_split.csv`).
3. Refit head on ISIC train ∪ non-ISIC train; threshold on ISIC val ∪ non-ISIC val.
4. Evaluate on ISIC test and non-ISIC test, separately and combined; write result doc.
5. If success criteria are met: replace `checkpoints/referral_head/referral_head.json`
   the same way the original training script does, and note the
   `contract_version`/behavior are unchanged (same JSON shape, same
   `ReferralHead.load()` path) — this is a re-fit, not a schema change.
6. If not met: write up the negative result the same way the
   composition audit was, and stop rather than iterate.
