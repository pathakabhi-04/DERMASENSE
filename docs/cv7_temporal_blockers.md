# CV-7 Temporal ("What Changed?") — Blockers

**Status:** Blocker 1 RESOLVED (2026-09-01) — see
`docs/cv7_temporal_rag_integration_spec.md` for the task definition and
the locked CV↔RAG-chatbot integration boundary, decided by the user.
Blocker 2 is in progress (plan decided, not yet executed). This
document is kept for the historical record of why CV-7 was blocked and
what unblocked it; the current design lives in the spec above.

## Why this wasn't a spec (historical)

Every other CV-phase spec in this project (`docs/cv1_5_router_spec.md`,
`docs/cv1_cv4_assembly_spec.md`, `docs/cv4_domain_evidence_spec.md`,
`docs/cv6_uncertainty_spec.md`) was written against two things already
in hand: a defined task (what does the component output?) and inspectable
data (what does the input actually look like?). CV-7 had neither at the
time this document was first written.

## Blocker 1: no task definition existed — RESOLVED

`docs/project_state.md` named the dataset ("UQ Longitudinal, 35,909
dermoscopic images, 7,038 lesions, 340 participants, 2–7 time points")
and the label "What Changed?", but nowhere in `docs/CV_MODEL_ARCHITECTURE_v1.0.md`,
`docs/CV_DATASET_SPEC_v1.0.md`, or anywhere else did an actual output
definition exist. Candidates considered at the time: a changed/stable
classification, a growth-rate estimate, new-lesion detection, or some
combination.

**Resolved 2026-09-01**: CV-7 produces a structured, numeric temporal
verdict (`STABLE | GROWING | SHRINKING | CHANGED_COLOR | NO_PRIOR_DATA`
+ magnitude + per-feature deltas + confidence), feeding CV-8 alongside
CV-4/CV-6. Full definition and the exact output contract:
`docs/cv7_temporal_rag_integration_spec.md`.

## Blocker 2: the data itself is not locally inspectable

Confirmed via a full filesystem search (`data/raw/`, `data/splits/`, and
the whole repo tree) — UQ Longitudinal has zero local footprint. Not the
images, not a metadata CSV, not a manifest. "On network volume" means
literally nothing about its schema, image format, or pairing structure
can be checked without a pod/volume session.

This is one level more blocked than CV-1.5's Stage 2 classifier was:
that spec was fully written and committed *before* touching RunPod,
because PAD-UFES and iToBoS were both already local. CV-7 can't even
reach that starting point without a session against the volume first.

## What CV-7 does have going for it, once unblocked

`lesion_id`/`lesion_uid`/`operational_lesion_uid` columns already exist
as schema fields across `src/data/dataset.py`, `src/data/manifest.py`,
and multiple split CSVs — identity-tracking scaffolding that a
same-lesion-across-time pairing mechanism could reuse, even though no
code currently groups rows by it. Also already tracked: the
iToBoS/UQ participant-level overlap caveat (`docs/project_state.md`,
deferred items table) — "do NOT treat iToBoS and UQ Longitudinal as
jointly independent for a single reported evaluation claim until
resolved." Whatever CV-7 becomes, that caveat applies to it.

## Re-entry condition

1. ~~Task definition decided by the user~~ — **done**, see
   `docs/cv7_temporal_rag_integration_spec.md`.
2. ~~Data access~~ — **done, bounded** (2026-09-02): a 30-participant,
   5.12GB stratified sample is staged locally and on the RunPod volume.
   See `analysis/quality/cv7_temporal_data/result.md`. The full 331-
   participant/57.7GB longitudinal set is deferred pending a volume
   resize, not blocking — the sample is sufficient to write and
   prototype the technical spec against.

The technical CV-7 spec (model, feature extraction, training/eval plan)
gets written next, against this staged sample, following the same
committed-before-running discipline
as every other component in this project.

## Update (2026-09-02): technical spec written, both blockers now closed

`docs/cv7_temporal_technical_spec.md`. Written after actually inspecting
the staged sample (not assumed): the images carry a physical mm ruler
enabling classical size measurement, multiple cameras were used (so
calibration must be per-image), and real clinical `Diagnosis` ground
truth exists per lesion (constant across that lesion's visits — an
outcome label, not a per-visit-in-time one). All CV-7 blockers are
closed; implementation is the next step, not further data/task
clarification.
