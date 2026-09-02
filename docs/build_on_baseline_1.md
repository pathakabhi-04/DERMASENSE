# Build on Baseline v1

**Status:** Planning document. Nothing in this file is implemented by
writing it — every item below is either a design decision waiting on
information this session doesn't have, or a documented weakness with
an explicit trigger condition, not a to-do list to work through
sequentially.

## How to read this document

Baseline v1 (`docs/project_state.md`, 2026-09-02) is a working
CV-1→CV-8 pipeline, verified end-to-end on real checkpoints and real
images. This document is what comes *after* that milestone: the live
delivery question, an honest weakness audit, and the CV-7 full-dataset
training question — three things explicitly asked for together,
because they share one discipline.

**The anti-rabbit-hole contract**, applied to every item in Sections B
and C: nothing here gets worked on because it "could be better." Each
item states (1) the issue, (2) the evidence it's real, (3) a concrete,
falsifiable **trigger condition** that would justify acting, and (4) a
**bounded first step** if that trigger fires — never an open-ended
"investigate further." An item without a met trigger stays exactly
where it is: documented, not started. This mirrors the discipline
already applied throughout this project (CV-1.5's Stage-1-before-Stage-2
gate, CV-7's own Stage-1-sufficiency finding, the SCC/BCC investigation
that was explicitly closed rather than re-opened without new evidence).

---

## A. Live feed: delivering CV-8's output to the RAG system

**Current state.** `RiskAssessment.to_dict()` (`src/risk/convergence.py`)
produces the locked contract in-process. `docs/cv8_sample_outputs/`
delivers 5 real examples as a static file. **No live delivery
mechanism exists or has been decided.**

### Prerequisite unknowns — answer these before building anything

These aren't technical unknowns this session can resolve by writing
code; they're product/infra decisions that determine which mechanism
is even sensible:

1. Does DermaSense already have a backend/hosting decision (a web
   app, mobile backend, serverless functions), or is this the first
   piece of server-side infrastructure for the product?
2. Does the RAG collaborator's system expect a **synchronous** call
   (user uploads → waits a few seconds → chat responds), or can it
   work **asynchronously** (a queue/webhook, with the chat surfacing
   the result whenever it arrives)?
3. What's the expected volume — one request at a time (a single user
   testing), or many concurrent users? This changes whether "one
   process holding models in memory" is even adequate.
4. Who owns the lesion-history store (which prior image to pass in)?
   That's a separate, already-flagged open item (`project_state.md`),
   but it determines whether the live-feed endpoint needs to accept a
   `prior_image_bgr` from the caller (current design) or look one up
   itself (would require this service to own that persistence too).

**Do not guess at these.** A live-feed design built on wrong
assumptions about sync/async or ownership is exactly the kind of
rework this document exists to prevent.

### Recommended default, if no constraint says otherwise

A minimal **synchronous HTTP endpoint** wrapping `DermaSensePipeline.predict()`:
one process, models loaded once at startup (not per request — loading
5+ checkpoints per call would make every request multi-second before
any inference even starts), a single `POST /assess` accepting an image
plus the optional `lesion_id`/`prior_image_bgr`/timestamps `predict()`
already takes, returning the same JSON `docs/cv8_sample_outputs/`
already demonstrates. This is the smallest thing that could plausibly
work for a first integration test with the collaborator, not a
production architecture.

**Bounded first step** (only once question 1-2 above are answered in
favor of "yes, build a sync endpoint"):

1. A thin FastAPI (or equivalent) wrapper around the existing
   `DermaSensePipeline` — no new CV logic, purely a transport layer.
2. Add a `contract_version` field to the response (not currently in
   the locked schema) so the collaborator's parser can detect a
   breaking change later without guessing from field presence.
3. Smoke-test it against the same 5 real images already in
   `docs/cv8_sample_outputs/sample_outputs.json` — the response should
   match byte-for-byte (modulo the new version field).
4. Where it runs (the RunPod pod already holds the checkpoints, but
   whether it should also serve traffic is an infra decision, not a CV
   one) — flagged, not decided here.

**Explicitly not now** — each of these only becomes worth building
once real traffic or a real integration attempt demonstrates the need,
not preemptively: authentication, rate limiting, horizontal scaling,
retry/queue semantics for failed requests, response streaming, TLS/cert
management, request logging/observability. Building these before a
single real request has been exchanged with the collaborator would be
solving problems that don't exist yet.

**A genuine prerequisite this document surfaces, not invents**: Section
B item 14 (latency) needs answering before this section can be
finalized — if a `predict()` call with CV-5/CV-6 enabled takes several
seconds, that alone might force the async answer to question 2 above
regardless of what the collaborator would otherwise prefer.

---

## B. CV implementation weaknesses — audited, trigger-gated

Every issue below is already true of the shipped baseline; none of
these are new problems introduced by writing this document. Numbers
are cited from the exact files that measured them.

| # | Component | Issue | Evidence | Trigger to act | Bounded first step | Effort |
|---|---|---|---|---|---|---|
| 1 | CV-2 | Wide-field/TBP domain gap drives CV-4 out-of-domain misfires — 32.7% of assessed wide-field images escalate to `URGENT_EVALUATION`, 85% require review, from a 12.9% per-candidate high-risk rate (357 BCC/39 MEL/1 SCC of 3,082 candidates) | `analysis/product_eval/cv1_cv4_assembly/result.md` | **Already CLOSED**, not open — tiling was the last permitted experiment under CV-2's own max-2-attempts stopping rule (`project_state.md`) and is deprioritized. Re-open ONLY if the product commits to wide-field capture as first-class (currently the deliberate assumption is zoomed single-lesion photos) | A genuinely new wide-field-aware detector architecture, not another tiling variant | High (blocked on a product decision, not a CV one) |
| 2 | CV-3 | Segmentation quality is domain-dependent: 78% pass rate (39/50, borderline fail vs. an 80% gate) on iToBoS wide-field crops vs. 95% clean-mask rate on UQ dermoscopic images | `analysis/quality/cv3_domain_itobos/result.md`; `analysis/quality/cv7_temporal_data/measurement_result.md` | Only if #1 is ever re-opened (wide-field TBP crops become first-class) | Assess whether a labeled TBP mask set can even be obtained/annotated — that's step zero, before any fine-tuning is considered | Medium, contingent on #1 |
| 3 | CV-4 | SCC/BCC representation overlap: centroid separation ratio 0.287 (low), confirmed not fixable by reweighting/SupCon alone | `analysis/scc_bcc/scc_bcc_feature_geometry.txt`; `project_state.md` | Already investigated and closed as a dead end. Re-open ONLY if a genuinely new signal is proposed (not another reweighting/loss-function variant on the same embeddings) | N/A until a new candidate signal exists | N/A |
| 4 | CV-4 | Out-of-domain crop-quality correlation is disclosed (`LOW_CROP_CONTRAST` flag), not corrected: BCC crop_contrast mean 0.208 (58.7% below 0.20), ACK mean 0.165, vs. MEL mean 0.337 (5.1% below 0.20) | `docs/cv4_domain_evidence_spec.md` | Only if a real-usage audit shows this flag alone fails to prevent bad outcomes at a meaningful rate (not a hypothetical) | Measure `risk_category` distribution conditioned on `LOW_CROP_CONTRAST` on a real sample; decide then whether a capture-guidance-style re-take prompt is justified | Low to measure, medium if a gate is built |
| 5 | CV-4/CV-8 | `native_class` taxonomy mismatch: pipeline emits PAD-UFES's 6 classes, the locked contract's own example shows ISIC's 8 | `docs/cv7_temporal_rag_integration_spec.md`; `docs/cv8_sample_outputs/README.md` | **Effectively already triggered** — this breaks the moment the collaborator writes taxonomy-dependent parsing code | A decision, not a CV task: either adopt PAD-UFES's 6 classes as the product taxonomy (update the contract's example) or define an explicit 6→8 mapping and document unmapped classes | Low (one decision + doc update) |
| 6 | CV-5 | `gradcam_mask_iou` has no calibrated threshold and isn't consumed anywhere yet — recorded, opt-in, currently inert | `src/risk/convergence.py` module docstring | Only once a suspected failure mode exists that it could explain (e.g. a URGENT_EVALUATION false-positive review shows attention on hair/background, not lesion) | Reuse the calibration.py/measurement.py pattern exactly: seeded real sample, real percentile, never a guessed cutoff | Low once triggered |
| 7 | CV-6 | Ensemble is "the cheapest possible version" — 2 checkpoints, same architecture/recipe, differing only by seed; captures training-stochasticity variance, not real model diversity | `docs/cv6_uncertainty_spec.md` | Only if a downstream audit finds high-confidence wrong predictions slipping past review at a meaningful rate | Add one ResNet50-backbone checkpoint (already exists as `SharedResNet50Backbone`) as a cheap architecture-diverse third member; re-measure `ensemble_agree` before going further | Low to try, evaluate before expanding |
| 8 | CV-7 | **Structural**: real users will essentially never have a physical ruler in frame, so `per_feature_deltas.size` will be `null` in ~100% of real production comparisons, not just the measured 0.3% double-confident rate on research data | `analysis/quality/cv7_temporal_data/delta_calibration_result.md` | **Not a bug — a permanent fact about the size channel**, not "fixable" by more CV work at all | Only a product decision (a reference-object capture protocol, e.g. asking for a coin/card in frame) would change this; not a CV task | N/A (product decision, not engineering) |
| 9 | CV-7 | `GROWTH_PCT_THRESHOLD`/`GROWTH_ABS_MM_FLOOR` are provisional placeholders, never calibrated (only 1 double-confident pair existed in the 300-pair sample) | `src/temporal/delta.py`; `delta_calibration_result.md` | Only if #8's product decision ever happens, making size deltas common enough to calibrate meaningfully. **Recommendation: do not pursue a full-dataset pull to fix this now** — see Section C | See Section C | N/A until #8 changes |
| 10 | CV-7 | Color-delta channel is confounded by cross-visit lighting/camera differences, not purely biological signal; the calibrated threshold is a conservative mitigation, not a fix | `delta_calibration_result.md` | Only if real usage shows `CHANGED_COLOR` false-alarming on benign lighting changes at a meaningful rate | Explore same-session color normalization against a fixed in-frame reference, if one is ever reliably present — untested, not attempted now | Medium, speculative until triggered |
| 11 | CV-7 | Stage 1's clinical-signal evaluation used one visit pair per lesion (not full history), n=89 scored malignant pairs — real and significant, not a large-n validation | `analysis/quality/cv7_temporal_data/stage1_evaluation_result.md` | Only if CV-8 starts weighting this signal heavily (primary rather than corroborating) | Re-run the SAME evaluation using ALL visit pairs per lesion in the ALREADY-staged data (no new pull) — cheapest possible expansion | Low (reuses existing script + data) |
| 12 | CV-8 | Escalation thresholds (`magnitude>=1.0`, `confidence>=2/3`) were chosen for defensibility, not tuned via a precision/recall-style sweep | `src/risk/convergence.py` | Only if real usage or a targeted re-evaluation shows the ratchet is miscalibrated (too aggressive or too lax) | Adapt `scripts/evaluate_cv7_stage1.py`'s Fisher/Mann-Whitney harness to test *final risk_category after escalation* vs. outcome, instead of raw verdict — infrastructure already exists | Low (adapts existing script) |
| 13 | All | Every number in this project comes from research datasets (PAD-UFES/ISIC/iToBoS/UQ Longitudinal) — zero validation on real DermaSense user photos, since there are no real users yet | (cross-cutting; no single citation) | Fires automatically the moment even a small real pilot batch of user photos exists | Re-run the same battery already built for every dataset transition this session: CV-1 rejection rate, CV-1.5 routing distribution, CV-3 mask degeneracy, CV-4 class/confidence distribution — on the pilot sample | Low (reuses existing scripts) |
| 14 | All | No latency/throughput measurement exists for the assembled pipeline, especially with CV-5/CV-6 enabled (a real backward pass / extra forward passes) | (not previously measured) | **Prerequisite for Section A**, not independently optional | Time `predict()` over the 5-10 images already in `docs/cv8_sample_outputs/`, with and without CV-5/CV-6, on the actual target hardware — a ~10-minute measurement | Low |

**Deliberately excluded from this table**: anything without a
concrete trigger condition someone could point to and say "yes, that
happened." A weakness with no trigger is not actionable — it's a
statement of general dissatisfaction, exactly the shape of experiment
that produced the SCC/BCC rabbit hole.

---

## C. CV-7 on the full UQ Longitudinal dataset

**Current state: not planned, not triggered.** Stage 1 (the classical,
deterministic pipeline) was evaluated against real malignant-outcome
data and found sufficient (`stage1_evaluation_result.md`: p=0.0135
non-STABLE rate, p=8.2×10⁻⁸ magnitude, both significant in the
expected direction). No Stage 2 model exists, and none is being built
on the basis of anything currently known. This section exists to
answer "what would it take, and when," not to schedule it.

### Two different things "train CV-7 on the full dataset" could mean

**(i) Recalibrating Stage 1's constants with more data — not model
training at all.** This is cheap in principle (no GPU, reuses existing
scripts) but has a specific, bad cost/benefit ratio worth stating
explicitly: `GROWTH_PCT_THRESHOLD` needs double-confident (both-visit
ruler-calibrated) pairs, which occur at ~0.3% — reaching even 30 usable
pairs for a rough percentile estimate would mean processing roughly
10,000 pairs, i.e. nearly the entire remaining dataset, to calibrate a
constant that (per item 8 above) will almost never be exercised by
real users anyway, since real photos have no ruler at all.
**Recommendation: do not do this.** The structural real-world ceiling
makes the target low-value regardless of sample size — this is
precisely the "more data might help" reasoning the anti-rabbit-hole
discipline exists to catch before it starts.

**(ii) Building and training a NEW learned Stage 2 model** (e.g. a
Siamese/paired-embedding network over image pairs, per the original
mention in `docs/cv7_temporal_technical_spec.md`) — a real ML project
requiring the full dataset, GPU time, and its own evaluation criteria.
**This is what "training CV-7" should be understood to mean if it ever
happens.**

### Trigger conditions for (ii) — any one, stated concretely, not "it might be better"

- **a.** A future, larger version of the Stage-1-style evaluation (item
  11 above, or item 12's escalation-specific version) finds a
  *specific* class of malignant change Stage 1's classical
  measurements cannot discriminate at all — not "accuracy could be
  higher," a named, demonstrated blind spot.
- **b.** The product commits to a reference-object capture protocol
  (item 8), making the size channel viable at scale, AND a bounded
  feasibility study (on paper, using already-staged data, before any
  training run) shows a learned scorer could plausibly use combined
  size+border+color+raw-pixel signal beyond what fixed percentile
  thresholds capture.
- **c.** An external requirement (regulatory/clinical validation)
  mandates a documented, reproducible model-based approach over
  threshold heuristics — a possible non-technical trigger, noted here
  but not something to speculate on further until it's real.

**None of these are currently met.**

### If a trigger fires: the concrete resource plan (pointer-level only)

This is deliberately not a full technical spec — writing one now,
before any trigger has fired, would itself be a rabbit hole. If/when
triggered:

- **Data**: full 331 longitudinal participants / 57.7GB, vs. the
  currently staged 84 participants / ~16GB. Requires the
  previously-costed RunPod volume resize (~70-80GB more, ~$5-6/month)
  — local and RunPod free space are already committed to the current
  staged subset.
- **Staging infrastructure already exists and needs no rework**: the
  same `zipfile`-based extraction + S3 upload + byte-for-byte
  verification pattern used twice this session (`analysis/quality/cv7_temporal_data/result.md`)
  — just re-pointed at the remaining ~274 participants.
- **Architecture**: a Siamese/paired-embedding network is the
  direction named in the original spec; the actual architecture,
  loss, and input representation get designed only once triggered,
  following the same committed-before-running discipline as every
  other component this session (a written spec, reviewed, before any
  training code).
- **Evaluation criteria, pre-registered before training starts**:
  must beat Stage 1's already-measured discrimination (p=0.0135
  non-STABLE rate, p=8.2×10⁻⁸ magnitude) on a held-out split — not
  "does it converge" or "does the loss go down."
- **Cost is not the primary gate.** A GPU RunPod session for this
  scale of data is likely single-digit hours plus the small monthly
  resize cost — cheap relative to model complexity/maintenance risk.
  The real gate is whether Stage 1 has been shown insufficient, not
  whether the compute is affordable.

### What not to do

- Do not pull the full 331-participant dataset "just in case" or "to
  have more data available."
- Do not start Stage 2 architecture experiments speculatively before
  a trigger fires.
- Do not treat "a learned model would probably do better" as a
  trigger by itself — that is the exact reasoning already rejected for
  CV-1.5's Stage 2 gate and for reopening SCC/BCC, applied here for
  consistency.
