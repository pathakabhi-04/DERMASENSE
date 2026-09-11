# Raising CV metrics toward deployment level

**Date:** 2026-09-12
**Gates this serves:** `docs/product_scope_and_readiness.md` §4
**Single number that matters:** end-to-end melanoma sensitivity.

Every step below is ordered by expected effect per unit of work, and
each carries a pre-committed decision rule so it can be stopped.

---

## 0. The binding constraint, stated once

| | MEL train | MEL test |
|---|---:|---:|
| PAD-UFES — what CV-4 was transferred to | **37** | **9** |
| ISIC2019 — already on disk | **3,256** | **666** |

CV-4 was pretrained on ISIC then transferred to PAD-UFES. PAD-UFES has
**37 melanoma training images**. The transfer optimised Macro-F1 across
six classes on a set where melanoma is 2.3% of the data — so the model
had almost no incentive to keep whatever melanoma capability ISIC
pretraining gave it, and every incentive to spend capacity on BCC (36%)
and ACK (32%).

**MEL recall 0.6667 is 6 of 9 images.** Its 95% CI is roughly 30–93%.
Before anything else: *we do not currently have a usable measurement of
melanoma recall.*

---

## Step 1 — Measure melanoma recall on a set big enough to mean something

**Cost:** hours. **Blocks everything else.**

Evaluate the current checkpoint on the **ISIC2019 test split (666
melanomas)**, not PAD-UFES's 9. Report sensitivity with a confidence
interval.

*Pre-committed rule:* if MEL sensitivity on 666 images is below 0.90,
treat every downstream metric as provisional and go to Step 2. If it is
above 0.90, the PAD-UFES transfer is the problem rather than the model,
which changes the plan — re-scope then.

Do this **before** retraining anything. Right now we would not be able
to tell whether a change helped.

---

## ⚠️ Step 2 as originally written is largely REFUTED by prior work

Re-reading the repo's own experiment record (2026-09-12) closed most of
what this step proposed. Recording it here so the GPU spend is not made
twice:

| Proposed | Status | Evidence |
|---|---|---|
| Class-weighted loss | **already tried, rejected** | `experiments/isic2019_resnet50_weighted.md`: val +0.0099 but frozen test 0.576770 → 0.575637, "did not demonstrate a meaningful improvement in held-out generalization" |
| Weight melanoma up | **misconceived** | ISIC weighting *down*-weights MEL (0.4457) — MEL is 17.7% of ISIC, not rare |
| More capacity | **already tried, no effect on MEL** | MEL recall: resnet18 0.5646, resnet50 0.5405, resnet50-weighted 0.5661 |
| Reweighting / SupCon on the overlap | **already tried, rejected** | `project_state.md`: "representation overlap confirmed, not fixable by reweighting/SupCon alone. Accepted as a known limitation." |
| Threshold on malignant probability | **already settled, and beaten** | `phase4_safety_policy/`: reviewing all `MONITOR` catches 100% of dangerous failures at 18.8% review rate; probability thresholds reach 43% at best |

Also corrected: the shipped safety gate routes every `MONITOR` to
`REVIEW` (`src/risk/safety_gate.py:67`), so melanomas landing in
`MONITOR` reach a human, not the user. See
`docs/product_scope_and_readiness.md` §1.

**What remains untried**, and is therefore where any GPU spend belongs:

1. **A malignant/benign head.** Nothing in any doc attempts it. The
   product asks "does this need a doctor?", which is a strictly easier
   question than the 6-way one, and the 6-way objective is what the
   evidence says is misaligned.
2. **Domain-appropriate data.** Every number in this repo is dermoscopy
   or clinical photography. None is a phone photo.

**Test before training:** of the 78 melanomas receiving <5% malignant
probability (`analysis/quality/mel_sensitivity/threshold_vs_retraining.md`),
does a binary head separate them? Extracted features already exist in
`analysis/scc_bcc/*.npz`, so this is answerable with no training at all.
If a binary probe cannot separate them either, the information is absent
from the representation and no head will fix it — which would redirect
the effort to data rather than objective.

---

## Step 2 (original) — Retrain with melanoma actually represented

**Cost:** days. **Expected effect: the largest available.**

The data is already downloaded. Three things to change, in order:

1. **Train on ISIC2019 as the primary set** (18,402 train / 3,256 MEL),
   with PAD-UFES as a domain-adaptation set rather than the target.
2. **Optimise for melanoma sensitivity at a fixed review rate**, not
   Macro-F1. Macro-F1 treats a missed melanoma and a misfiled seborrheic
   keratosis as equally bad. They are not.
3. **Class-weighted or focal loss**, with the weight derived from the
   real prevalence rather than guessed. A weighted ResNet50 experiment
   already exists (`docs/experiments/isic2019_resnet50_weighted.md`) —
   start from what it found rather than from scratch.

*Pre-committed rule:* target MEL sensitivity ≥0.90 on the ISIC2019 test
split at a review rate the product can absorb. If three training runs
do not clear 0.85, the problem is not loss weighting and the plan needs
rethinking — stop and reassess rather than running a fourth.

---

## Step 3 — Fix detection recall (0.8098, "AT FLOOR")

**Cost:** days. **Effect: multiplies whatever Step 2 achieves.**

~19% of lesion-containing images yield no candidate, and a lesion never
detected can never be classified. This caps end-to-end sensitivity no
matter how good CV-4 becomes: at 0.81 detection, a perfect classifier
still yields 0.81 end-to-end.

`cv2_status.md` already records that YOLO11s capacity made no difference
(+0.0009), so the ceiling is not model size. Investigate recall at lower
confidence thresholds, and accept more false positives — a false
positive costs a review, a false negative costs a melanoma.

*Pre-committed rule:* detection recall ≥0.90 at a false-positive rate
the review workflow can absorb. Measure both; do not optimise recall
alone.

---

## Step 4 — Measure end-to-end, not per-stage

**Cost:** hours. **Do this after every change from here on.**

The per-stage numbers are misleading because they compound and are
measured on different sets. What matters is: *of N images containing a
melanoma, how many produce an output that sends the user to a doctor?*

That is the only number that describes the product. Build it as a
standing evaluation and run it on every checkpoint.

Note that "sends the user to a doctor" is the right success criterion —
not "classified as MEL". A melanoma classified as BCC still produces
`URGENT_EVALUATION`, and that is a good outcome. A melanoma classified
as NEV produces `MONITOR`, and that is the failure that matters.

---

## Step 5 — Prospective validation on real phone photos

**Cost:** weeks, and it needs data that does not exist yet.

PAD-UFES is clinical photography; ISIC is dermoscopy. Neither is a
phone camera in bad light at an awkward angle. Every number above is
measured on the training distribution and will drop on real input —
possibly a lot.

This is the step that decides whether the product works, and it cannot
be skipped or simulated. Plan for collecting a real-world set early;
it has the longest lead time of anything here.

---

## What NOT to do

- **Don't chase Macro-F1.** It is the metric that produced a model
  missing a third of melanomas while looking respectable.
- **Don't tune on PAD-UFES's 9 test melanomas.** Any change will look
  significant and none will be.
- **Don't add architecture complexity before Step 2.** The ResNet18 vs
  ResNet50 experiments already show capacity is not the constraint;
  data balance is.
- **Don't buy GPU for deployment yet.** CPU inference is 0.5–2.5s, which
  is adequate. GPU is a training cost, not a serving one, at this scale.
- **Don't revisit the RAG layer.** It is ahead of the CV side and
  waiting. Both its gates pass.

---

## Ordering summary

| Step | Cost | Effect | Gate |
|---|---|---|---|
| 1. Measure MEL recall properly (n=666) | hours | none directly — makes the rest measurable | CI reported |
| 2. Retrain with melanoma represented | days | **largest** | MEL sens ≥0.90 |
| 3. Detection recall 0.81 → 0.90 | days | multiplies step 2 | recall ≥0.90 |
| 4. End-to-end evaluation | hours | reveals compounding | standing metric |
| 5. Real phone-photo validation | weeks | decides viability | honest drop measured |

Step 1 first. We currently cannot tell whether any change helps.
