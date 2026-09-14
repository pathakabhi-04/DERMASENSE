# The data constraint: what is missing, and what would close it

**Date:** 2026-09-15
**Status:** specification. This is the thing four independent
investigations converged on, written down so it can be acted on or
explicitly declined rather than rediscovered a fifth time.

## 1. The finding this rests on

| investigation | held-out clinical AUC | conclusion |
|---|---:|---|
| linear probe on broader data | 0.73 | missed its 0.80 bar |
| backbone fine-tune (1 clinical domain) | 0.63–0.67 | seen domains fine, unseen not |
| routing / detection attribution | — | neither is the lever; 40% of melanomas still missed with both removed |
| multi-domain (3 domains + per-domain heads) | 0.63–0.64 | domain count is not the constraint either |

In-domain the system is fine: **0.9024 melanoma routing on ISIC**. It
degrades only when the capture source changes. Every architectural lever
tried has failed to close that, and the failures were consistent,
pre-registered, and cheap.

**The constraint is not model capacity, training objective, detection
recall, routing, or the number of training domains. It is that we have no
labelled images from the distribution the product would actually run on.**

## 2. What we have

| dataset | n | capture | labels | patient IDs |
|---|---:|---|---|---|
| ISIC2019 | 25,331 | dermoscopy | 8-class, biopsy-backed | lesion IDs |
| PAD-UFES-20 | 2,298 | smartphone clinical (Brazil) | 6-class, biopsy-backed | **yes** |
| DDI / DDI-2 | 1,973 | clinical (Stanford) | free-text, biopsy-proven | DDI-2 only |
| Fitzpatrick17k (atlas subset) | 583 | teaching atlas | diagnosis strings | no |
| iToBoS | 8,481 | wide-field TBP rig | **no diagnosis labels** | — |

Total melanomas outside ISIC: **107**, and PAD-UFES contributes 9 to its
own test split.

## 3. What is missing, specifically

Not "more data". Five concrete gaps, in priority order.

### 3.1 Images from the deployment capture path — the binding gap

Every clinical source above is a **clinician photographing a patient**:
controlled lighting, a steady hand, a known working distance, often a
dermatology clinic's camera. The product's input is **a member of the
public photographing their own skin** with a phone, one-handed, in
bathroom lighting, at whatever distance feels natural.

No dataset in the table is that. PAD-UFES is the closest and is still
clinic-captured. This is the gap that four experiments have been
indirectly measuring.

**Minimum useful quantity**, from the counts this project has been
working with: ~100 melanomas is enough to *measure* a rate to ±10%
(`external_6class_result.md`); training a source-adapted head needed
roughly 400–1,600 images per domain to reach 0.93 in-domain
(`cv4b_backbone_finetune_result.md`, Fitzpatrick from 437 images). So:

- **to measure**: ~100 biopsy-confirmed melanomas + ~500 benign, phone-captured
- **to adapt**: ~500–1,000 labelled phone images spanning the class mix

### 3.2 Ground truth that survives contact with a pathologist

Fitzpatrick17k's labels are atlas captions, not biopsy reports. The
mapping onto PAD's six classes required judgment calls documented in
`build_external_6class_eval.py` — `porokeratosis actinic` is not actinic
keratosis, Bowen's may or may not be SCC. **1,108 of 1,900 external
images had no PAD analogue at all and were discarded.**

Any new collection needs **biopsy- or dermatologist-confirmed labels in a
fixed taxonomy agreed in advance**, not free text to be mapped later.

### 3.3 Skin tone coverage, recorded rather than assumed

DDI exists precisely because dermatology datasets under-represent darker
skin. DDI and DDI-2 carry Fitzpatrick skin type; ISIC and PAD-UFES
largely do not. The project has never reported per-skin-tone performance
because most of its data cannot support it.

A collection that does not record Fitzpatrick type prospectively cannot
answer the one equity question this product will certainly be asked.

### 3.4 Patient identifiers

DDI has none. Fitzpatrick has none. This project found **25 of 132
DDI-2 test images shared a patient with training** and could only fix it
because DDI-2 happens to carry `deidentified_patient_id`
(`cv4b_retrain_result.md`). For DDI and Fitzpatrick the same leak may
exist and is undetectable.

**Patient ID is not optional metadata.** Without it, no split can be
trusted and every held-out number is potentially inflated.

### 3.5 Paired longitudinal images

CV-7 compares the same lesion across visits and is, by the readiness
doc's own assessment, the strongest part of the system. Its ruler
calibration achieves confident coverage on **~4%** of cases. There is no
dataset here with the same lesion photographed twice by the same user
weeks apart — so the component with the best product story is the one
with the least data behind it.

## 4. What a sufficient collection looks like

Minimum viable, to *measure* rather than to train:

| requirement | target |
|---|---|
| capture | user's own phone, unsupervised, their own lighting |
| volume | ~600 lesions, ~100 of them melanoma |
| labels | biopsy or dermatologist consensus, fixed taxonomy |
| metadata | patient ID, Fitzpatrick type, body site, device model |
| longitudinal | a subset re-photographed at ≥8 weeks, for CV-7 |
| governance | ethics approval, consent covering model training, data agreement |

**This is not a dataset that can be scraped or downloaded.** It requires
a clinical partner with a biopsy pathway, because the labels come from
pathology and the melanomas come from people who actually have melanoma.

## 5. Honest cost, and the alternative

Lead time is months, not weeks: ethics approval, a recruiting clinic, and
the base-rate problem — melanoma is rare, so ~100 confirmed melanomas
means a large recruited cohort or a partner who already sees them.

**If that runway does not exist, the correct move is not to approximate
it.** Three experiments already show that substituting other people's
clinical photographs does not work. The alternative is to ship the parts
of the product that do not depend on classification at all
(`product_scope_and_readiness.md` §2) and report the classification work
as the measured negative result it is.

## 6. What this spec does NOT justify

- Another architecture, loss, or augmentation experiment. The constraint
  has been isolated four times.
- Acquiring a fifth public clinical dataset. The failure generalised
  across every one tried; a fourth source of the same kind is predicted
  to behave like the other three, and that prediction is itself now a
  finding rather than a guess.
- Shipping a diagnosis claim on the strength of in-domain numbers.
