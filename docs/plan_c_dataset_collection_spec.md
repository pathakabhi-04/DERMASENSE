# Plan C: building the deployment dataset

**Date:** 2026-09-15
**Status:** specification, not yet started. Plan A is what ships now.
**Prerequisite reading:** `data_constraint_spec.md` (what is missing and
why), `plan_a_narrow_product_spec.md` (what ships meanwhile).

## 1. The central idea: Plan A is the collection instrument

The dataset that cannot be downloaded is one where **ordinary people
photograph their own skin on their own phones, and a pathology result is
attached afterwards.** That is precisely what the narrow product
generates in the course of being useful.

```
user photographs a lesion  ->  app stores photo + measurement + quality
        |                                   (Plan A does this already)
        v
user shows the summary to a GP  ->  GP refers  ->  biopsy
        |                                              |
        +--------- linkage code -----------------------+
                                                       v
                                        label attached to the ORIGINAL
                                        user-captured photo
```

Nothing else in this spec matters as much as that loop. It turns a
months-long data-acquisition project into a by-product of shipping, and
it is the only mechanism proposed here that produces images from the
actual deployment capture path.

**It has to be designed in from the first release**, because consent and
the linkage identifier cannot be added retroactively to photos already
taken.

## 2. What we are collecting

| field | source | why |
|---|---|---|
| original image, full resolution | user's phone | the deployment capture path |
| device model, OS | phone | measure whether capture quality tracks hardware |
| capture quality signals | CV-1 | the honest distribution of real-world photos |
| lesion outline + area fraction | CV-3 | the signal that predicts failures |
| **patient identifier** | account | **non-negotiable — see §5** |
| body site | user tap on body map | confounder, and clinically relevant |
| **Fitzpatrick skin type** | user self-report, validated subset | the equity question |
| repeat photos of the same lesion | app | the only path to CV-7 validation |
| **biopsy / dermatologist label** | clinical partner | the actual ground truth |
| date of capture and of biopsy | both | time-to-diagnosis, and staleness |

## 3. The capture protocol is "no protocol"

This is the part most easily got wrong.

> **The photo must be taken the way a user would take it, unsupervised,
> in their own lighting, with no coaching beyond what the shipped app
> already does.**

Every dataset that failed us — PAD-UFES, DDI, DDI-2, Fitzpatrick — is a
*clinician* photographing a patient under clinic conditions. If a nurse
stands over the user and says "hold it steadier, come closer", we will
have spent months rebuilding the dataset we already have and already
know does not transfer.

The app's own capture prompts are allowed and required, because they
ship with the product and are therefore part of the deployment
distribution.

## 4. Where the labels come from

Two routes, in order of value:

1. **Biopsy-confirmed** (gold). User → GP → dermatology → excision →
   histopathology. Slow, definitive, and the only acceptable label for
   melanoma.
2. **Dermatologist consensus** (silver). Two independent
   dermatologists reviewing the *user's own photo*; disagreements
   adjudicated by a third. Cheaper, faster, weaker — and adequate for
   the benign classes that will never be biopsied.

**The taxonomy is fixed before collection starts**, in writing, as the
PAD-UFES six (`ACK, BCC, MEL, NEV, SCC, SEK`) plus an explicit
`OTHER`/`UNCERTAIN` bucket. The alternative is what happened with
Fitzpatrick17k: free-text labels mapped afterwards by judgment, with
1,108 of 1,900 images discarded for having no analogue.

**Benign lesions will not be biopsied.** Nobody excises an obvious
seborrheic keratosis to label a dataset. So the benign classes come from
route 2 and that asymmetry must be recorded per-image
(`label_source: biopsy | consensus`), not averaged away.

## 5. Split discipline, designed in rather than retrofitted

This project found **25 of 132 DDI-2 test images shared a patient with
training**, and could only detect it because DDI-2 happens to carry a
patient identifier. DDI and Fitzpatrick carry none, so the same leak may
exist there and is permanently undetectable.

Therefore:

- **Every image carries a patient ID from the moment of capture.** Not
  optional metadata. Without it no split can be trusted and every
  held-out number is potentially inflated.
- **Splits are grouped by patient**, using the machinery already built
  (`build_cv4b_splits.py`).
- **A held-out test set is frozen before any modelling**, and a
  *second* held-out set is reserved and never touched until a final
  pre-deployment evaluation. The first will be contaminated by
  iteration; assume it.

## 6. How much, and how we will know when we have enough

From this project's own measurements, not from a rule of thumb:

| purpose | requirement | basis |
|---|---|---|
| **measure** a melanoma routing rate to ±10% | ~100 melanomas | `external_6class_result.md`: 107 melanomas gave CI [0.53, 0.72] |
| **adapt** a head to this domain | 500–1,000 labelled images | `cv4b_backbone_finetune_result.md`: Fitzpatrick reached 0.93 in-domain from 437 |
| **validate CV-7** | ~200 lesions photographed ≥2× at ≥8 weeks | no existing data at all |

Melanoma is rare — roughly 1 in 100 suspicious lesions presented in
primary care. **~100 melanomas implies on the order of 10,000 submitted
lesions**, or a partner who already sees a melanoma-enriched population
(a pigmented-lesion clinic, a mole-mapping service). The recruitment
arithmetic, not the technology, sets the timeline.

## 7. Governance — the part with the longest lead time

- Ethics/IRB approval covering **model training**, not merely service
  provision. Consent that permits care but not training is the most
  common and most expensive mistake here.
- Explicit consent for **retention and secondary use**, separable from
  app usage, and revocable.
- Data-sharing agreement with the clinical partner covering who holds
  the images, who can train on them, and what happens on termination.
- Pseudonymised identifiers only; body-site photos can be identifying,
  and faces/tattoos must be handled explicitly.
- A stated position on whether the dataset will ever be released. It
  should be — the field's whole problem is dataset scarcity — but that
  has to be in the consent from day one.

## 8. Milestones, with a kill criterion at each

| # | milestone | gate to continue |
|---|---|---|
| 1 | Clinical partner agreed, ethics submitted | a partner with a biopsy pathway and melanoma volume |
| 2 | Plan A shipping with consent + linkage code | linkage demonstrably works end-to-end on 10 cases |
| 3 | 500 lesions, any label | capture-quality distribution differs measurably from PAD-UFES — if it does not, we are collecting clinic photos again |
| 4 | 100 confirmed melanomas | — |
| 5 | Frozen test split, first honest measurement | melanoma routing on **real deployment photos**, reported with CI |
| 6 | Adaptation attempt | only now, and only with the discipline of the four prior experiments |

**Milestone 3's gate is the important one.** It is the early check that
this collection is not quietly reproducing the failure mode we already
measured, and it can be evaluated after a few hundred images rather than
after a year.

## 9. Risks, honestly

- **Recruitment shortfall.** Most likely failure. 10,000 lesions for 100
  melanomas is a large funnel, and a partner with melanoma volume is the
  single highest-leverage thing to secure first.
- **Label lag.** Biopsy results arrive weeks after capture; the linkage
  has to survive that, and users churn.
- **Selection bias.** People who download a mole app and then see a
  doctor are not the general population. Record it; do not pretend
  otherwise.
- **The photos may be worse than expected.** That is the point of
  collecting them, and the result will be uncomfortable.
- **It may not close the gap.** Four experiments narrowed the constraint
  to data; none of them proves that data alone fixes it. This is the
  best-supported hypothesis available, not a guarantee.

## 10. What this spec does not authorise

- Starting modelling before milestone 5.
- Substituting another public clinical dataset for milestone 3. That
  substitution has now failed four times and its failure is a finding.
- Shipping any diagnosis claim on the strength of interim numbers.
