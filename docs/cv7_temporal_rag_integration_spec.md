# CV-7 Temporal ("What Changed?") + RAG Integration — Product/Architecture Spec

**Status:** Task definition and the CV↔RAG integration boundary are
LOCKED (2026-09-01, product decision by the user). This resolves
blocker 1 from `docs/cv7_temporal_blockers.md`. Blocker 2 (data access)
is in progress, not resolved — see "What's still open" below. This
document is therefore a **product/architecture spec**, not yet the
technical CV-7 spec (model, features, training plan) — that still needs
UQ Longitudinal's actual schema, which isn't inspectable until the data
lands on the pod, per this project's committed-before-running
discipline (the technical spec gets written once, against real data,
same as every other component).

## What CV-7 computes

Given two images of the same lesion at different times, CV-7 produces a
structured, numeric verdict describing what changed — magnitude,
direction, and per-feature deltas (size, color, border) — with a
confidence. CV-8 consumes this alongside CV-4's diagnosis and CV-6's
uncertainty (same convergence pattern as everything else in this
project) to decide whether the change pattern reads as natural/benign
(tanning, minor fluctuation) or concerning (rapid growth, irregular
color/border evolution — the evolution signal dermatologists already
use clinically).

## Why the RAG chatbot is strictly downstream, not CV-7 itself

This was a real fork in the design, decided deliberately rather than by
default, using the same dependency-direction principle that shaped
every other boundary in this project (CV-3's mask doesn't gate CV-4's
input; the classifier doesn't decide risk category; see
`docs/cv5_cv6_evidence_architecture.md` for the pattern stated in full).

**The rejected alternative:** let an LLM narrate "what changed" directly
from the two raw images or their embeddings. This was rejected because
it would make the temporal-change claim *generated*, not *computed* —
unit-testable and regression-testable the way every other CV component
in this project is, versus a confident, plausible-sounding claim with no
verified ground truth behind it. An LLM asserting "this lesion appears
to have grown" without a deterministic measurement backing it is a
hallucination risk sitting directly upstream of a risk decision — the
exact failure mode this project has spent the most effort eliminating
elsewhere (see the CV-1 recalibration and CV-4 domain-evidence work:
both were about not trusting a plausible-looking signal without
verifying it against ground truth first).

**The design that was chosen:** CV-7 stays a classical, structured-output
component. The RAG chatbot's role is retrieval-augmented *explanation*
of facts CV-7/CV-8 have already computed and verified — it can enrich
its narration with general dermatology context, glossary definitions,
educational content, but it narrates CV-7's number, it does not invent
one.

## Architecture: two diagrams

**Diagram 1 — the RAG pipeline itself (the collaborator's system, no
DermaSense-internal coupling):**

```
                              USER QUESTION
                    ("why is this flagged?",
                     "what does asymmetry mean?")
                                   |
                                   v
                    +------------------------+
                    |    Query encoder /      |
                    |    embedding model      |
                    +------------------------+
                                   |
                                   v
                    +------------------------+
                    |    Vector store /       |
                    |    knowledge base       |
                    |  (dermatology facts,    |
                    |   glossary, FAQ,        |
                    |   educational content)  |
                    +------------------------+
                                   |
                                   v
                          retrieved passages
                                   |
                                   v
                    +------------------------+
                    |    Prompt assembly      |
                    |  (retrieved context +   |
                    |   user question +       |
                    |   [session context])    |<---- see Diagram 2:
                    +------------------------+       this slot is where
                                   |                  CV-8's output enters
                                   v
                    +------------------------+
                    |    LLM generation       |
                    +------------------------+
                                   |
                                   v
                    +------------------------+
                    |  Safety / guardrail     |
                    |  check (no diagnosis    |
                    |  invented, no claims     |
                    |  beyond retrieved +      |
                    |  provided context)       |
                    +------------------------+
                                   |
                                   v
                          USER-FACING ANSWER
```

**Diagram 2 — the integration point: CV-8's output feeding the RAG
pipeline:**

```
   DermaSense CV pipeline                     RAG system (collaborator's)
   (everything upstream of this               (everything downstream of
    line is untouched by RAG)                  this line is untouched by CV)
   -------------------------------------------------------------------------
                                        |
   CV-4 diagnosis   ---+                |
                        |               |
   CV-6 uncertainty ---+--> CV-8 -------+---> structured JSON  +----------+
                        |    risk           contract (below)   | Session  |
   CV-7 temporal    ---+    engine          -------------->    | context  |
   (structured verdict)                                        | injector |
                                        |                       +----------+
                                        |                             |
                                        |                             v
                                        |                  +------------------+
                                        |                  |  Prompt assembly  |
                                        |                  | (session context +|
                                        |                  |  retrieved        |
                                        |                  |  passages)        |
                                        |                  +------------------+
                                        |                             |
                                        |                             v
                                        |                    ... rest of Diagram 1
                                        |                       (LLM, guardrail,
                                        |                        user answer)
   -------------------------------------------------------------------------

   ONE-WAY DEPENDENCY: RAG reads CV-8's JSON. CV-7/CV-8 have no knowledge
   of the RAG system, its retrieval store, or its embedding model.
```

**The dependency runs one way.** RAG depends on CV-8's structured
output; CV-7/CV-8 have zero awareness the RAG system exists. This means
the RAG pipeline can be rebuilt, swapped, or completely redesigned
without ever touching `src/temporal/` or `src/risk/`, and CV-7 can be
retrained without breaking the RAG system, as long as the contract's
shape doesn't change. This is the same independent-evaluability property
this project has protected at every other boundary (CV-1/CV-1.5/CV-2/
CV-3/CV-4 can each be re-evaluated in isolation; this extends that
property across the CV/RAG system boundary too).

## The handoff contract

This is the actual deliverable — the piece worth nailing down precisely
before the collaborator writes code against it, since it is the whole
interface between two independently-built systems:

```json
{
  "lesion_id": "string",
  "diagnosis": {
    "native_class": "MEL | BCC | SCC | AK | NV | BKL | DF | VASC",
    "probabilities": { "...": 0.0 }
  },
  "risk_category": "LOW | MEDIUM | HIGH",
  "risk_reason": "short machine-generated justification string, not LLM-authored",
  "temporal": {
    "verdict": "STABLE | GROWING | SHRINKING | CHANGED_COLOR | NO_PRIOR_DATA",
    "magnitude": 0.0,
    "confidence": 0.0,
    "per_feature_deltas": { "size": 0.0, "border": 0.0, "color": 0.0 },
    "compared_timestamps": ["...", "..."]
  },
  "uncertainty": {
    "confidence": 0.0,
    "requires_review": true
  },
  "quality_flags": ["..."]
}
```

`risk_reason` being explicitly "machine-generated, not LLM-authored" is
the contract's load-bearing line — it's what keeps the safety-relevant
justification inside the verified, testable system, with the RAG layer
free to elaborate on it in natural language without ever being the
source of the claim.

## Two discrepancies against the current codebase — one resolved

1. **RESOLVED (2026-09-02) in `src/risk/convergence.py`.**
   `risk_category` is kept as a NEW field, separate from
   `ProductAction` — not unified with it. `ProductAction` remains the
   internal, authoritative action, used unchanged by the existing
   safety gate. `risk_category` is derived from it for this external
   contract only: `URGENT_EVALUATION -> HIGH`, `EVALUATE_SOON ->
   MEDIUM`, `MONITOR -> LOW`, `UNKNOWN -> HIGH` (fail-safe, mirroring
   `safety_gate.py`'s own fail-to-REVIEW handling of UNKNOWN). See
   `src/risk/convergence.py`'s module docstring for the full reasoning,
   including how CV-7's verdict is allowed to escalate this category
   (a one-way ratchet, never a de-escalation).
2. **Still open, deliberately left unresolved by the CV-8 work above.**
   `native_class` enum (`MEL | BCC | SCC | AK | NV | BKL | DF | VASC`)
   is the ISIC2019 8-class taxonomy
   (`src/models/native_classifier.py::ISIC2019_CLASSES`), while the
   assembled pipeline (`src/inference/orchestrator.py`) uses the
   PAD-UFES 6-class head exclusively. `src/risk/convergence.py` passes
   through whatever `CandidateResult.predicted_class` actually is
   (PAD-UFES) rather than fabricating a mapping to the contract's
   8-class example — reconciling the taxonomies is a CV-4/data
   question, orthogonal to wiring CV-7's signal into risk convergence,
   and is not resolved here.

## What's still open

- **Blocker 2 (data access) — in progress, not resolved.** Plan:
  download UQ Longitudinal locally → upload to the RunPod S3 bucket as
  a zip → extract on the pod → train. Same pattern already used for
  CV-1.5 Stage 2 (`docs/cv1_5_router_spec.md`) and the CV-1.5 held-out
  iToBoS test subset. Not yet executed as of this document.
- **The technical CV-7 spec** (model architecture, feature extraction
  for size/color/border deltas, training approach, evaluation criteria)
  is not written here, deliberately — it needs UQ Longitudinal's actual
  schema (image format, pairing structure, whether lesion identity
  across time points is directly usable via the existing
  `lesion_id`/`lesion_uid` columns noted in
  `docs/cv7_temporal_blockers.md`). Gets written once the data is on
  the pod and inspectable, following the same committed-before-running
  discipline as every other spec in this project.
- **CV-8's convergence implementation is DONE for CV-4+CV-7** (2026-09-02):
  `src/risk/convergence.py::assess_risk()` emits this exact contract,
  taking a `CandidateResult` (CV-4+CV-6 evidence) and an optional
  `TemporalResult` (CV-7). `src/risk/action_mapping.py` and
  `src/risk/safety_gate.py` are unchanged and still used internally.
  **Not yet wired**: the real pairing logic that finds a user's prior
  visit image and produces the `TemporalResult` to pass in
  (`TemporalPipeline.assess_pair` takes two already-selected images);
  `assess_risk()` is called per-candidate manually for now, not
  threaded through `DermaSensePipeline.predict()` itself.
