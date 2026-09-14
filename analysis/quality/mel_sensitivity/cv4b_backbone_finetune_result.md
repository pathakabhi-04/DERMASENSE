# CV-4b backbone fine-tune: adaptation does not reach an unseen source

**Date:** 2026-09-14
**Design:** `docs/cv4b_backbone_finetune_design.md` (pre-committed; §8 fixed
these criteria and the meaning of each outcome *before* the run)
**Raw results:** `cv4b_finetune/{pooled,loso_atlas,loso_stanford}_result.json`
plus per-epoch `_history.json`
**Checkpoints:** `checkpoints/cv4b_finetune/<fold>/best.pt` (gitignored)

## Verdict: do not ship. The ISIC-only head stays in production.

| fold | test set | non-ISIC AUC | benign referred | MEL routed | ISIC AUC | §8 |
|---|---|---:|---:|---:|---:|:--|
| pooled | held-out images, **seen** sources | 0.8497 | 0.3273 | 0.9545 (21/22) | 0.9180 | **pass** |
| loso_atlas | Fitzpatrick, **unseen** | 0.6682 | 0.6567 | 0.6988 (58/83) | 0.9162 | fail |
| loso_stanford | DDI+DDI-2, **unseen** | 0.6257 | 0.5836 | 0.5200 (13/25) | 0.9162 | fail |

This is **exactly** the outcome §8 pre-registered as its third branch:

> *LOSO < 0.80 but pooled ≥ 0.80 → adaptation works for seen sources and
> does not generalise to unseen ones. [...] it would mean per-deployment-
> source calibration data is required, and that no amount of training on
> these three sources makes a user's phone camera safe by itself.*

Deployment is a fourth unseen source. The LOSO folds are the number that
describes the product; `pooled` is not.

## 1. The fine-tune did not beat doing nothing on an unseen source

Against the un-fine-tuned ISIC-only head on the *same* test sets
(`domain_shift_second_source.md`):

| unseen source | metric | baseline (no fine-tune) | fine-tuned | change |
|---|---|---:|---:|---:|
| Fitzpatrick (n=583, 83 MEL) | AUC | 0.6924 | 0.6682 | **−0.024** |
| Fitzpatrick | MEL routed | 0.7952 [0.70,0.87] | 0.6988 [0.59,0.79] | **−0.096** |
| DDI+DDI-2 (n=1317, 25 MEL) | AUC | ~0.601 | 0.6257 | +0.025 |
| DDI+DDI-2 | MEL routed | ~0.757 [0.57,0.89] | 0.5200 [0.33,0.70] | **−0.237** |

**Melanoma routing — the safety-relevant metric — fell on both unseen
sources.** The Wilson intervals overlap, so no single comparison is
decisive at these melanoma counts; what makes it worth acting on is that
the direction is consistent across two independent held-out sources, and
that the better-powered AUC comparison on Fitzpatrick (283 malignant)
also moved the wrong way.

Spending GPU on backbone adaptation bought nothing here that the frozen
ISIC-only head did not already have.

## 2. The clearest single number: melanomas ranked below benign lesions

On `loso_stanford`:

```
benign referred : 0.5836
melanoma routed : 0.5200   <- LOWER than the benign rate
```

The model refers 58% of benign lesions and catches 52% of melanomas. A
score carrying *no* melanoma information would route melanomas at
roughly the benign rate. This one is below it. Malignant-vs-benign AUC
is 0.6257, which is above chance — but that is driven by non-melanoma
malignancies; melanoma-specific discrimination on DDI/DDI-2 is
effectively absent.

## 3. The operating point does not transfer either

The threshold is chosen on validation for a 40% benign-referral budget.
On the unseen source it actually spends **58.4%** (`loso_stanford`) and
**65.7%** (`loso_atlas`). Even setting aside ranking quality, a
threshold calibrated on one source does not hold on another — so a
deployment could not be configured from source-agnostic validation data
even if the ranking were adequate.

## 4. The fold that "passed" is inflated by pooling

`pooled` cleared the gate on **combined** non-ISIC AUC (0.8497). Per
source, on the same fold:

| source | n | malignant | AUC |
|---|---:|---:|---:|
| ddi | 131 | 34 (26%) | 0.7122 |
| ddi2 | 132 | 14 (11%) | 0.7088 |
| fitzpatrick17k | 116 | 56 (48%) | 0.9265 |
| n-weighted mean of the three | | | **0.7766** |
| **reported combined** | 379 | 104 | **0.8497** |

Combined AUC exceeds the per-source mean by **+0.073**. Pooling sources
whose malignant prevalence ranges from 11% to 48%, and whose score
distributions differ, inflates AUC through between-source separation
that has nothing to do with within-source diagnosis. **Only Fitzpatrick
individually clears 0.80.**

That is a flaw in the gate metric I pre-committed, not a flaw introduced
after seeing the result — recording it because it means "pooled passed"
is weaker than it looks, and any future version of this experiment
should gate on per-source AUC or a prevalence-matched pool.

`pooled`'s headline melanoma routing (0.9545) also rests on **22
melanomas, with DDI-2 contributing zero** — the design already flagged
this fold's melanoma routing as not a usable headline number, which is
why it is not cited as evidence of anything here.

## 5. What did work

- **No catastrophic forgetting.** ISIC test AUC is 0.9162–0.9180 across
  all three folds, against 0.9050 for the shipped head. The 50/50
  oversampling mix (§6) did its job: clinical photos at ~6% of the data
  drove real gradient signal without destroying the in-domain task.
- **Per-source adaptation is genuinely effective.** Fitzpatrick reaches
  0.9265 AUC in `pooled`, and `loso_stanford` reached 0.9643 val AUC on
  held-out Fitzpatrick after training on only 437 Fitzpatrick images.
  Given labelled data *from a source*, this recipe adapts to that source
  well.

The problem is specifically **zero-shot transfer to a source not in
training** — which is what deployment is.

## 6. An asymmetry that sharpens the still-open question

In-domain validation, each LOSO fold trained and validated on its own
source group:

- train Stanford → Stanford val: **~0.70**
- train Fitzpatrick → Fitzpatrick val: **~0.96**

DDI/DDI-2 are hard to separate malignant-from-benign *even when the
model trains on them*. That is not a transfer effect. It reframes the
question left open by `domain_composition_audit.md`: DDI's difficulty is
intrinsic, not merely distributional, and any future investigation
should ask what makes DDI/DDI-2 lesions hard rather than what makes them
*different*.

## 7. Decision, per the rule set in advance

- **Nothing ships.** `checkpoints/referral_head/referral_head.json` and
  the shipped backbone are unchanged. No production path was touched at
  any point in this experiment.
- **Do not escalate to full-backbone fine-tuning.** §12 ruled this out in
  advance on 1,140 clinical images, and nothing here argues for it: the
  failure is not that layer4 lacked capacity, it is that the target
  (an unseen capture source) is not reachable from this training data.
- **Do not iterate on hyperparameters.** §12: "one configuration,
  pre-committed." The result is informative, not ambiguous.
- **The actionable path is labelled data from the deployment source**,
  not more generic training. That is a product and data-collection
  decision, not a modelling one, and it should be raised as such.

## Caveats

- 25 melanomas in `loso_stanford` and 22 in `pooled` are small; every
  melanoma-routing figure carries a wide Wilson interval, quoted above.
  The malignant-vs-benign AUCs (283–525 malignant) are better powered.
- Two source groups is the minimum for leave-one-source-out. Two folds
  agreeing is suggestive, not conclusive, about a fourth unseen source.
- `loso_atlas` selected its checkpoint at epoch 3 and early-stopped at
  10; its best validation AUC was 0.7139, so that fold was weak before
  test was ever touched.
