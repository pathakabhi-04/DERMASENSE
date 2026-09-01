# CV-7 Temporal ("What Changed?") — Blockers

**Status:** Not spec'd. This document records what's blocking a spec,
not a spec itself — there isn't enough locked to pre-commit criteria
against yet.

## Why this isn't a spec

Every other CV-phase spec in this project (`docs/cv1_5_router_spec.md`,
`docs/cv1_cv4_assembly_spec.md`, `docs/cv4_domain_evidence_spec.md`,
`docs/cv6_uncertainty_spec.md`) was written against two things already
in hand: a defined task (what does the component output?) and inspectable
data (what does the input actually look like?). CV-7 has neither.

## Blocker 1: no task definition exists

`docs/project_state.md` names the dataset ("UQ Longitudinal, 35,909
dermoscopic images, 7,038 lesions, 340 participants, 2–7 time points")
and the label "What Changed?", but nowhere in `docs/CV_MODEL_ARCHITECTURE_v1.0.md`,
`docs/CV_DATASET_SPEC_v1.0.md`, or anywhere else does an actual output
definition exist. This is a genuine open product question, not a
technical gap — candidates include:

- A binary/multi-class **changed vs. stable** classification per lesion
  pair.
- A continuous **growth-rate estimate** (area/diameter change over
  time).
- **New-lesion detection** — did a lesion appear that wasn't there at
  the prior visit?
- Some combination of the above.

Each implies a different label structure, a different model shape, and
a different evaluation metric. None can be chosen from the codebase
alone — this needs a product decision.

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

Both of the following, in either order:

1. **Task definition decided by the user** — which output CV-7 should
   produce (see the candidates above).
2. **Data access** — a pod/volume session (or a volume-to-S3 metadata
   sync, following the pattern already used for CV-1.5's held-out
   iToBoS test subset) to inspect UQ Longitudinal's actual schema,
   image format, and pairing structure.

Once both are available, this becomes a normal spec-writing task,
following the same committed-before-running discipline as every other
component.
