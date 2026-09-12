# DermaSense — what to ship now, and what "deployment-ready" would mean

**Date:** 2026-09-12
**Purpose:** separate what the models actually support today from what
the product eventually wants to be, so the near-term build is honest and
the long-term bar is explicit.

---

## 1. The blunt version

The engineering is in good shape. The classifier is not.

| Stage | Metric | Value | Source |
|---|---|---|---|
| CV-2 detection | image-level recall | 0.8098 (target ≥0.90) | `cv2_status.md` |
| CV-4 classification | Macro-F1 (ISIC) | 0.5756 | `project_state.md` |
| CV-4 classification | Macro-F1 (PAD-UFES) | 0.5996 | `project_state.md` |
| Phase 4 gate | **MEL recall** | **0.6667** | `phase4_safety_gate.md` |
| CV-7 ruler calibration | confident coverage | ~4% | `cv7_temporal_technical_spec.md` |

Those compound: ~19% of lesion-containing images produce no candidate at
all, and of the melanomas that *are* detected roughly one in three is
misclassified. End-to-end melanoma sensitivity lands near **50–55%**.

And that 0.6667 is **6 of 9 test images** — PAD-UFES has 9 melanomas in
its test split. The 95% confidence interval is roughly 30–93%. We do not
actually know the melanoma recall; we know it is somewhere in a range
that includes "very bad".

### Why this matters more than it sounds — with an important correction

Trace a melanoma misread as `NEV` through `src/risk/action_mapping.py`:

```
MEL photographed -> classified NEV -> MONITOR -> risk_category LOW
```

**Correction (2026-09-12).** An earlier draft of this section said `LOW`
"reaches the user unchallenged". That is wrong about the shipped system
and the error is worth recording rather than quietly deleting.

`src/risk/safety_gate.py:67` routes **every** `MONITOR` action to
`REVIEW`, unconditionally. That policy was chosen deliberately in Phase
4 after comparing alternatives, and the comparison
(`analysis/product_eval/phase4_safety_policy/`) shows why:

| policy | review rate | dangerous failures caught |
|---|---:|---:|
| `low_risk_high_risk_prob_ge_0.10` | 4.5% | 43% |
| `global_confidence_lt_0.90` | 67.6% | 57% |
| **`low_risk_prediction_review`** (shipped) | **18.8%** | **100%** |

Reviewing every `MONITOR` catches 100% of high-risk→MONITOR errors *by
construction*, at a lower review rate than any confidence-threshold
policy achieving even 57%. The design is sound and already settled.

**So the real exposure is narrower and different in kind.** It is not
"the model tells melanoma patients they are fine". It is:

> ~28% of melanomas land in a review queue, and the product is only as
> safe as that queue actually being staffed and worked through.

That is an operational and staffing dependency, not a model defect. It
becomes a model defect only if `MONITOR` is ever auto-released — which
is precisely the change an unsupervised consumer app would be tempted to
make, and must not.

**A reassuring false negative is the one output that can kill someone.**
It is strictly worse than no app: it can stop a person seeing a doctor
they would otherwise have seen. The gate exists so that output cannot be
produced without a human first looking.

---

## 2. What to ship now: the narrow product

There is a genuinely useful product here that does **not** depend on the
classifier being right. Every piece already exists and already works.

### 2.1 Capture assistance — CV-1, already good

Blur, contrast, framing and quality scoring are implemented, calibrated
against real data, and honest about their own limits. Bad photographs
are the single biggest obstacle to any later assessment, and helping a
user take a good one is valuable on its own.

*Ships as:* live capture feedback — "too blurry", "move closer", "the
lesion is cut off at the edge".

### 2.2 Longitudinal tracking with change flags — CV-3 + CV-7

Change detection does not require knowing *what* the lesion is. CV-7
compares two photos of the same lesion and reports `STABLE`, `GROWING`,
`CHANGED_COLOR` with calibrated thresholds, a one-way escalation ratchet
that never de-escalates, and `NO_PRIOR_DATA` as a first-class honest
outcome. The `/assess` endpoint makes it callable and the measurement
token makes repeat visits cheap.

*Ships as:* a photo diary that says "this looks different from three
months ago — worth showing someone", never "this is benign".

**This is the strongest part of the system**, and it is being
under-sold by being bundled with a diagnosis claim it cannot support.

### 2.3 "Here's what to show your dermatologist" — CV-5 + RAG

Grad-CAM overlays, the quality flags, the photo timeline, and the RAG
layer's grounded, cited explanations of general dermatology. A one-page
summary a patient can hand to a clinician.

The RAG layer is genuinely ready: Phase 1 and Phase 2 gates pass, every
answer cites real sources, it refuses to assert diagnoses, and it now
says plainly when the corpus doesn't cover a question.

### 2.4 The rule that makes it honest

> **The narrow product never emits reassurance.** Its worst output is
> "we can't tell — see a clinician". It never says low risk, benign,
> or probably fine.

A system with 55% sensitivity can be honest about uncertainty. It cannot
be reassuring. Drop the `LOW` risk label from anything user-facing until
the sensitivity supports it.

**Restated precisely, given §1's correction.** The shipped gate already
prevents `MONITOR` from being auto-released, so the rule is not asking
for a new safety mechanism — it is asking that the existing one never be
relaxed, and that the *narrow* product simply not have a reassuring
output to relax toward:

- **Never auto-release `MONITOR`.** The gate does this today. An
  unsupervised deployment is the scenario that breaks it.
- **If there is no reviewer, there is no `LOW`.** A product without a
  review queue must show "we can't tell — see a clinician" for what
  would have been `MONITOR`, not silence and not reassurance.

That second point is what makes the narrow product shippable without a
clinical staffing commitment: capture assistance, change tracking and a
clinician summary have no reassuring output to begin with, so they do
not depend on a review queue existing.

---

## 3. As an academic project

The narrow scope is *better* academically, not a retreat:

- **A real, measured contract boundary** between CV and RAG, with the
  calibrated-vs-raw confidence bug found and fixed by measurement.
- **A safety layer with numbers behind every threshold** — 0.12 Jaccard,
  0.25 containment, 0.45 low-similarity, all calibrated against real
  distributions rather than chosen, each with its limits stated.
- **Negative results that are genuinely interesting**: 25% of an
  authoritative medical corpus tripping a naive banned-phrase rule; a
  hedge-scoping bug that passed every metric while missing 8 of 8
  adversarial diagnosis claims; an uncovered question returning
  skin-cancer chemotherapy instructions.
- **An honest sensitivity analysis** that concludes the system is not
  deployable, and says why with data.

That last point is the strongest thing in the project. Most student
work overclaims; measuring your own system and reporting that it misses
half of melanomas is a better result than a number nobody checked.

---

## 4. The bar for the real product

Four gates, in order. None is optional.

1. **Melanoma sensitivity.** Triage/rule-out devices in this space are
   typically held to >95% sensitivity. This is the whole ballgame —
   nothing else matters until it moves. See
   `docs/cv_metrics_improvement_plan.md`.
2. **Prospective validation on real phone photos.** PAD-UFES clinical
   images and ISIC dermoscopy are not what a user's camera produces.
   Current numbers will not survive that shift, and validating on the
   training distribution proves nothing about deployment. **No longer
   just asserted** — measured directionally on biopsy-proven clinical
   photography (DDI/DDI-2, 25 melanomas), and the finding is broader
   than CV-4b alone: CV-4b's malignant-vs-benign AUC drops from 0.9050
   (ISIC dermoscopy) to 0.5742, near chance, but the shipped 6-class
   classifier ALSO degrades on this same data — melanoma routing
   0.7180 (ISIC, in-domain) → 0.3200 (clinical photos), worse than
   either in-domain figure or CV-4b on the identical images. This
   implicates the shared representation, not just one linear head.
   See `analysis/quality/mel_sensitivity/referral_head_on_ddi.md` and
   `native_classifier_on_ddi.md`.
3. **No reassurance, ever** (§2.4), until sensitivity supports it.
4. **Regulatory.** Anything returning a risk category on a skin lesion
   is Software as a Medical Device. FDA/CE has a defined pathway; it
   needs planning early, not after launch.

---

## 5. What this does not change

The architecture, the contract, the safety discipline and the
integration work all stand. Fail-loud parsing, calibrated confidence,
one-way escalation, `NO_PRIOR_DATA` as a first-class outcome, null
never coerced to zero — that is the right foundation for a medical
product.

It just cannot compensate for a classifier trained on 37 melanomas.
