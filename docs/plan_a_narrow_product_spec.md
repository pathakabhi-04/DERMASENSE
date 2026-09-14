# Plan A: the narrow product — what changes from the original plan

**Date:** 2026-09-15
**Status:** **this is what ships.** Decision taken 2026-09-15 on the
evidence in `data_constraint_spec.md`.
**Companion:** `plan_c_dataset_collection_spec.md` — Plan A is also the
instrument that collects Plan C's dataset.

## 1. The one-sentence change

> The original plan shipped a **risk verdict** about a lesion. Plan A
> ships **capture quality, change over time, and a clinician handoff**,
> and never emits a verdict at all.

Everything below follows from that. Nothing is deleted; the diagnosis
path keeps running, it just stops reaching the user.

## 2. What is different, component by component

| component | original plan | Plan A |
|---|---|---|
| CV-1 quality | convenience gate | **load-bearing** — primary user-facing output |
| CV-1.5 router | routes to CV-2 or CV-3 | unchanged |
| CV-2 detection | wide-field candidates | unchanged; known not to be the constraint |
| CV-3 segmentation | crop for CV-4 | **promoted** — outline + measurement is a product feature |
| CV-4 6-class | narrates the diagnosis | **runs, logged, never surfaced** (§4) |
| CV-4b referral head | routing input to CV-8 | **runs, logged, never surfaced** (§4) |
| CV-5 Grad-CAM | explains the diagnosis | explains *the outline*, for the clinician page |
| CV-6 uncertainty | gates the verdict | logged only; its confidence carries no error information |
| CV-7 temporal | supporting signal | **the headline feature** |
| CV-8 risk convergence | emits `risk_category` to the user | emits it **internally only**; no user-facing risk category |
| RAG | explains the diagnosis | explains dermatology in general and the change tracking (§5) |
| Safety gate | routes `MONITOR` to a review queue | **no queue needed** — nothing to auto-release |

## 3. What the user actually sees

1. **Before the shutter** — "too blurry", "move closer", "the lesion is
   cut off". Now the most valuable output in the product: dangerous
   misclassifications cluster on under-zoomed photos
   (`abstention_result.md`), and this is detectable *before* any
   classification runs.
2. **First capture** — outline, measurement (mm with a ruler in frame,
   otherwise pixels), stored. *"Saved — photograph this again in 3
   months."*
3. **Repeat capture** — `STABLE` / `GROWING` / `CHANGED_COLOR`, side by
   side, one-way escalation, `NO_PRIOR_DATA` when there is nothing to
   compare.
4. **Clinician page** — timeline, overlay, measurements, quality flags,
   on one page.
5. **Questions** — grounded, cited general dermatology; refuses what the
   corpus does not cover.

Worst-case output: *"This looks different from last time — worth showing
someone."* Latency unchanged at 0.5–2.5s CPU, no GPU.

## 4. The diagnosis path keeps running, silently — and why

CV-4 and CV-4b stay in the pipeline and their outputs are logged, never
rendered. Three reasons, and the third is the important one:

1. Removing them would be a large change to a tested pipeline for no
   product gain.
2. Plan B (dermoscopy) turns them back on with no re-integration.
3. **They are the baseline Plan C measures against.** Every logged
   prediction sitting next to a future biopsy result is a free
   evaluation datapoint on real deployment images — the measurement this
   project has never been able to make.

**Requirement:** the logged prediction must never be rendered, exported
to the clinician page, or passed to the RAG layer. §5 enforces the last
of these; the first two are UI obligations recorded here so they are not
lost.

## 5. The rule that makes it honest, restated for Plan A

> **The narrow product never emits reassurance.** Its worst output is
> "we can't tell — see a clinician". It never says low risk, benign, or
> probably fine, and it never names a diagnosis.

Concretely, none of these may reach a user surface:

- `native_class` or any class name
- `risk_category`, including `LOW`
- `confidence` or `calibrated_confidence` attached to a diagnosis
- any phrasing implying a lesion is harmless

Note this is *stronger* than the original §2.4, which only forbade
`LOW`. Plan A forbids the diagnosis label too, because a user told
"most likely class: NEV" has been reassured whether or not a risk
category was attached.

**What may be shown:** the outline, the measurement, the change verdict,
the quality flags, and the fact that an assessment was not possible.

## 6. Contract impact

`CV-8` keeps emitting its full v1.2 payload — the contract does not
change, the *consumption* does. The RAG layer's view of it is narrowed
in `src/rag/cv_context/schema.py` (see
`rag_plan_a_refactor.md`), and the UI obligation is stated in §5.

This deliberately avoids a v1.3: the contract is correct, well tested,
and Plan B would need the full payload back. Narrowing at the consumer
is reversible; narrowing the contract is not.

## 7. What Plan A must add that the original plan did not need

These are new work, not existing components:

1. **Consent and linkage for Plan C** (§1 of the collection spec) —
   patient identifier, retention consent, and a linkage code that
   survives to a biopsy result. **Cannot be retrofitted** to photos
   already captured, so it must be in the first release.
2. **A body-site tap** at capture — needed for Plan C, cheap now.
3. **Self-reported Fitzpatrick type**, optional, once — the equity
   question this project currently cannot answer for most of its data.
4. **Reminder scheduling** at 3 months — CV-7 is the headline feature
   and it does nothing without a second visit.
5. **Prediction logging** with the schema Plan C will join against.

## 8. What Plan A explicitly drops

- The review queue and its staffing model. Nothing auto-releases a
  reassurance, so no human backstop is required for safety. (A clinical
  reviewer may still be desirable; it is no longer a *safety*
  dependency.)
- Any regulatory claim resting on diagnosis. A product that measures and
  compares photographs, and never asserts a clinical conclusion, sits in
  a materially different category from Software as a Medical Device —
  **this needs a regulatory opinion, not an engineering assumption**,
  and it is flagged here rather than assumed.
- The `LOW` risk label, everywhere, permanently.

## 9. Success criteria

Plan A is not graded on melanoma sensitivity — it makes no such claim.
It is graded on:

| criterion | target |
|---|---|
| capture quality improves with prompts | measurable drop in low-area-fraction captures between first and second attempt |
| repeat-visit rate | ≥30% of users return at 3 months — below this CV-7 never fires and the product has no core |
| clinician page found useful | qualitative, from partner GPs |
| Plan C linkage works | ≥10 images linked end-to-end to a pathology result |
| no reassurance ever emitted | zero instances in review of shipped copy and logs |

The second row is the real risk. The product's best component only
functions on a return visit, and nothing in this project has yet
measured whether users come back.
