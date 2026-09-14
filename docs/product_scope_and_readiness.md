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

**Correction (2026-09-14): these do not compound the way this section
originally claimed.** The earlier text multiplied CV-2's ~19% miss rate
by CV-4's misclassification rate to land end-to-end sensitivity near
50–55%. Those two stages are **not on the same path**.

CV-1.5 routes each image by framing: pre-framed / lesion-centric photos
go **straight to CV-3, skipping CV-2 entirely**; only wide-field images
go through CV-2 (`docs/cv1_5_router_spec.md`). PAD-UFES — the only
*training* branch carrying diagnosis labels — is pre-framed by
construction, so CV-2's 0.8098 recall never gates it.

**Second correction, same day:** an earlier version of this paragraph
went further and said CV-2 "is not in the path" for close-up photos
generally. That was too strong, and the external evaluation below
refutes it. On 792 real clinical photographs from three sources,
**CV-1.5 routes 45% to wide_field** (DDI 42%, DDI-2 76%,
Fitzpatrick17k 21%), and CV-2 then returns no candidate for 30.8% of
those. CV-2 *is* on the critical path for roughly half of real-world
close-ups. The router's 1.000/1.000 was measured where dataset identity
stood in for framing; a third source is the first real test of it.

**So the end-to-end number is now measured rather than inferred**
(`analysis/quality/mel_sensitivity/external_6class_result.md`, 107
melanomas — against PAD-UFES test's 9):

| outcome for a melanoma | share |
|---|---:|
| reaches a clinician | **57.9%** |
| assessed → `MONITOR` | **34.6%** |
| never assessed | 7.5% |

Conditional on assessment, melanoma routing is 0.6263 [0.53, 0.72], and
6-class macro recall is 0.3117 against the in-domain 0.5996. The old
50–55% figure was arithmetic over stages that are not all on the same
path; 57.9% is the measurement, and it happens to land nearby for
different reasons.

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

## 1b. The decision this now forces (2026-09-15)

The classification gap has been isolated to a **data** constraint, not a
technical one — four pre-registered investigations, summarised in
`docs/data_constraint_spec.md`. That closes the "keep improving the
model" path and forces a product choice. There are exactly three honest
options.

| option | what it needs | what it gives up |
|---|---|---|
| **A — Narrow the product** (§2) | nothing; it is already built | any diagnosis claim |
| **B — Restrict the domain** | dermoscopy input; a clinician customer | the consumer market |
| **C — Collect deployment data** | clinical partner, ethics, months | time, and it may still fail |

**A — Narrow the product.** Capture assistance, longitudinal tracking,
and the clinician summary do not depend on the classifier at all. Every
piece exists and works today. §2 below is that product. **Cheapest, and
the only one that is honest right now.**

**B — Restrict the domain.** The system is not broken; it works where it
was trained — **0.9024 melanoma routing on ISIC dermoscopy**. A product
that requires a dermoscope attachment, or that targets a GP practice
rather than a consumer, is operating in-distribution. That is a real
option and a genuinely different product, with a different customer and
a different regulatory story.

**C — Collect deployment-source data.** The technically correct answer
and the one the evidence points at. It needs a clinical partner with a
biopsy pathway, ethics approval, and months of lead time, because the
labels come from pathology and the melanomas come from people who have
melanoma. `data_constraint_spec.md` specifies exactly what would be
sufficient. **Only viable if the project has that runway** — and if it
does not, approximating it with more public clinical datasets is already
shown not to work.

These are not exclusive: A ships now and funds the wait for C. What is
*not* available is a fourth option where more modelling closes the gap.

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

**Newly load-bearing (2026-09-15).** This is no longer just a
convenience. `abstention_result.md` found that the pipeline's *dangerous*
failures — melanomas assessed and sent to `MONITOR` — concentrate on
images where CV-3 measures a **small lesion relative to the frame**, i.e.
under-zoomed photographs. The signal is detectable **at capture time,
before any classification runs**, and carries a 1.62× lift over chance
(consistent across 98% of 40 random splits).

It was deliberately *not* made into a post-hoc abstention gate: as a gate
it caught only ~37% of dangerous misses while abstaining on 23% of
everything, which failed its pre-committed bar. As a **capture prompt**
it costs the user one retake, has no false-abstention cost, and gives
them something to act on. "Move closer" is the intervention.

Also settled there, and worth knowing: the classifier's own
`calibrated_confidence` is **0.7031 on melanomas it routes correctly and
0.7028 on the ones it misses** — it carries no information about its own
errors, so no confidence-keyed abstention is possible.

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

A system that routes **57.9%** of melanomas to a clinician on external
clinical photographs (`external_6class_result.md` — measured, not
inferred) can be honest about uncertainty. It cannot be reassuring. Drop
the `LOW` risk label from anything user-facing until the sensitivity
supports it.

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

### 3.1 The central contribution (2026-09-15)

The classification work is now a **pre-registered negative result**, and
it is a stronger contribution than a tuned number would have been:

> Across four independent experiments, the melanoma-routing gap on
> unseen capture sources was isolated to a **data** constraint. It is not
> model capacity, training objective, detection recall, routing, or the
> number of training domains.

| experiment | held-out clinical AUC | what it eliminated |
|---|---:|---|
| linear probe on broader data | 0.73 | head-fitting data |
| backbone fine-tune | 0.63–0.67 | frozen representation |
| routing / detection attribution | — | CV-1.5 and CV-2 |
| multi-domain + per-domain heads | 0.63–0.64 | domain count |

What makes it defensible rather than merely disappointing is the method:

- **decision rules fixed before each run**, so no bar moved after seeing
  a number;
- **leave-one-source-out**, because held-out *images* from a seen source
  overstate deployment — a distinction the results confirm (seen 0.85 vs
  unseen 0.63);
- **patient-grouped splits**, after finding 25 of 132 DDI-2 test images
  shared a patient with training;
- **confounds reported against our own interest** — `dg_pad` "passed" at
  0.8522 and was discarded because the starting checkpoint had already
  been fine-tuned on PAD-UFES;
- **corrections recorded, not overwritten** — the CV-2 compounding claim,
  the "router is misrouting" reading, the 0.7250 figure that came from
  silently dropping 45% of inputs.

Several of those are cases where the honest reading was *worse* than the
convenient one. That is the part worth writing up.

Two further negative results stand on their own: the classifier's
calibrated confidence carries **no** information about its own errors
(0.7031 vs 0.7028), and CV-2's 30.8% no-candidate rate on clinical
photographs costs almost no melanoma sensitivity because **75% of what it
drops is benign**.

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
   just asserted** — measured across three independent clinical-photo
   sources (DDI, DDI-2, Fitzpatrick17k; 108 melanomas combined). Two
   distinct findings, not one: CV-4b's (the referral head's)
   malignant-vs-benign AUC collapses on **every** source tested
   (0.59–0.69, vs 0.9050 on ISIC) — a general, reproducible transfer
   failure. The shipped 6-class classifier's failure is **not**
   general — severe on DDI/DDI-2 (melanoma routing 0.72→0.25–0.33) but
   much milder on Fitzpatrick17k (0.69, close to in-domain) — so
   source-specific capture/curation differences, not the shared
   backbone broadly, are the leading open question there. See
   `analysis/quality/mel_sensitivity/domain_shift_second_source.md`
   (start here) and the two single-source docs it supersedes in scope
   (`referral_head_on_ddi.md`, `native_classifier_on_ddi.md`).
   A follow-up refit of CV-4b on a broader (ISIC + DDI/DDI-2/Fitzpatrick17k)
   training distribution recovered real ground (non-ISIC AUC 0.60→0.73)
   but did not clear its own pre-committed bar (0.80) and increased the
   non-ISIC benign-referral cost to 54%, so it was **not** shipped — the
   ISIC-only head stays in production. See
   `analysis/quality/mel_sensitivity/cv4b_retrain_result.md`.
   **Backbone fine-tuning was then tried and also did not ship**
   (`cv4b_backbone_finetune_result.md`). With leave-one-source-out folds,
   adaptation reached held-out images from *seen* sources (0.85) but not
   an *unseen* source (0.63–0.67) — and melanoma routing on unseen
   sources fell *below* the un-fine-tuned baseline (0.80→0.70 on
   Fitzpatrick; ~0.76→0.52 on DDI/DDI-2, where melanomas were routed at a
   lower rate than benign lesions). ISIC held at 0.916, so this is not
   forgetting; it is that **zero-shot transfer to a capture source absent
   from training does not happen**, which is precisely what a user's phone
   is. A further **multi-domain run** — three training domains per fold
   plus per-domain auxiliary heads, adding PAD-UFES which the first
   fine-tune had omitted — did not change this
   (`cv4b_domain_generalization_result.md`): held-out-domain AUC 0.6334
   and 0.6385 against a pre-committed 0.75 bar, one of them *worse* than
   its predecessor. ISIC held at 0.91 throughout. The gate-2 conclusion is
   therefore stronger than "unvalidated": **per-deployment-source labelled
   calibration data is required**, and that is a data-collection
   commitment, not a modelling fix. Four independent investigations now
   converge on it.
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
