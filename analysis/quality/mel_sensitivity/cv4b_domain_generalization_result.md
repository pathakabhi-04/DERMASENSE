# Multi-domain training: hypothesis falsified, and the one "success" is confounded

**Date:** 2026-09-14
**Design:** `docs/cv4b_domain_generalization_design.md` (§6 fixed these
bars, and the meaning of each outcome, before the run)
**Raw:** `cv4b_finetune_dg_{atlas,stanford,pad}_result.json`, per-epoch
`cv4b_finetune/dg_*_history.json`

## Verdict: nothing ships, and this is the end of the domain-generalization line

| fold | held out | AUC | vs predecessor | benign referral | MEL routed | §6 |
|---|---|---:|---:|---:|---:|:--|
| `dg_atlas` | Fitzpatrick17k | **0.6334** | 0.6682 → **−0.0348** | 0.8133 | 0.7831 (83) | fail |
| `dg_stanford` | DDI + DDI-2 | **0.6385** | 0.6257 → **+0.0128** | 0.6227 | 0.6400 (25) | fail |
| `dg_pad` | PAD-UFES | 0.8522 | — | 0.2969 | 0.7778 (9) | ungated — **and confounded, see §3** |

The rule required **both** gated folds ≥ 0.75. One moved by +0.013
(noise), the other went **backwards**. Adding a third training domain and
per-domain auxiliary heads did not produce features that transfer to an
unseen capture source.

Per §6, pre-committed: *"Only the pooled/seen numbers improve again → the
constraint is per-source data... Stop. This is the last
domain-generalization attempt."* That is the branch this landed in.

## 1. The premise was right; the fix didn't help

The previous experiment genuinely was deficient — its LOSO folds trained
on **one** clinical domain each, and domain-invariant features cannot be
learned from a single domain. That criticism stands, and PAD-UFES really
had been omitted from the bundle.

Both were fixed. Every fold here trains on three domains, and the result
is unchanged. **So domain count was not the binding constraint.** A
correct criticism of an experiment does not imply the corrected
experiment will succeed, and that distinction is the whole value of
having run it.

## 2. `dg_atlas` is worse, and its melanoma number is an illusion

AUC fell from 0.6682 to 0.6334 on the identical test set. Its melanoma
routing of 0.7831 looks healthy until you read the line above it:
**benign referral 0.8133**. It refers 81% of benign Fitzpatrick lesions.
Routing 78% of melanomas while referring 81% of everything benign is not
discrimination — the melanoma rate is *below* the benign rate, the same
pathology `cv4b_backbone_finetune_result.md` found on DDI/DDI-2.

It also peaked at **epoch 2 of 9** and then flatlined into early
stopping. The auxiliary heads absorbed the signal quickly and the shared
backbone stopped moving.

## 3. `dg_pad` "passed", but PAD-UFES is not a held-out domain

This is the one result that looked like success, and it should not be
read as one.

The starting checkpoint is `pad_ufes_c1_partial_finetune_seed42_best.pt`.
Its own metadata:

```
experiment: C1
architecture: resnet50_layer4_finetune
dataset_id: pad_ufes
source_checkpoint: checkpoints/isic2019_resnet50_weighted_best.pt
```

Experiment C1 was *"partial ResNet-50 fine-tuning: ISIC → PAD-UFES"* —
**layer4 was already adapted to PAD-UFES before any of this started.**
Every fold in every one of these experiments begins from weights trained
on PAD-UFES train. Holding PAD out of the *fine-tune mix* does not hold
it out of the *backbone's history*.

So 0.8522 measures a domain the representation already knows, not
generalization to an unseen one. It is consistent with everything else
here — seen domains work, unseen domains do not — and it is emphatically
not a fourth data point in favour of the hypothesis.

This confound applies to `dg_pad` only; Fitzpatrick17k and DDI/DDI-2 are
genuinely absent from the backbone's training history.

## 4. What worked, again

ISIC test AUC is **0.9105–0.9147** across all three folds, against 0.9050
for the shipped head, with melanoma routing 0.89–0.90. No catastrophic
forgetting, for the third experiment running. The 50/50 oversampling, the
frozen-stage discipline, the per-domain heads, the resume path — the
machinery is sound. The hypothesis was wrong.

## 5. What this settles

Four independent investigations now converge on the same constraint:

| investigation | conclusion |
|---|---|
| linear probe refit | 0.60 → 0.73, missed its 0.80 bar |
| backbone fine-tune (1 clinical domain) | seen 0.85, unseen 0.63–0.67 |
| CV-1.5 routing / CV-2 attribution | neither is the lever; 40% of melanomas miss with both removed |
| **multi-domain (3 domains + per-domain heads)** | **unseen still 0.63–0.64** |

The gap is not architecture, not detection, not routing, and not the
number of training domains. **It is labelled data from the source you
intend to deploy on** — which is a data-collection commitment, not a
modelling problem, and should be raised as such.

## 6. Decision

- **Nothing ships.** `checkpoints/referral_head/referral_head.json` and
  the shipped backbone are untouched, as in every experiment in this
  series.
- **No DANN arm.** §3 of the design made it conditional on arm 1 clearing
  its bar. It did not.
- **No further domain-generalization attempts**, per §6 as written in
  advance.
- The honest prior stated in the design — *"the DG literature frequently
  finds that DG methods fail to beat plain pooled training"* — turned out
  to describe this result exactly.

## Caveats

- `dg_stanford` has 25 melanomas and `dg_pad` 9; their melanoma routing
  figures carry wide intervals and are not load-bearing. The AUCs
  (241–288 malignant) are better powered and are what the rule gates on.
- Two gated folds is the minimum for this design. Two failing folds plus
  one confounded pass is not proof that no multi-domain method could
  work — it is evidence that this one, at this scale, does not.
- `AUX_LOSS_WEIGHT = 0.5` was fixed in advance and never tuned. A
  different weight might behave differently; exploring that is exactly
  the hyperparameter search §9 refused.
