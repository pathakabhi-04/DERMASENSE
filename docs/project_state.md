# DermaSense — Project State Document

**Last updated:** CV-1 -> CV-4 pipeline assembly session (2026-09-01).
**Purpose:** bootstrap context for any new session. Read this before
touching any code. All decisions referenced here are committed in git
with rationale; this document is a navigation index, not a substitute
for those files.

---

## Product overview

DermaSense is a smartphone-first skin lesion risk triage system. It is
NOT a diagnostic tool — it triages ("monitor / get checked / urgent
referral") by combining CV evidence with a downstream risk engine.
The key product constraint that has shaped every CV decision: real users
submit phone photos, not dermoscopic images. Domain transfer from
dermoscopic training data to smartphone-clinical images is a first-class
concern throughout.

**Architecture (locked):**
```
INPUT IMAGE
     │
     ▼
CV-1  Quality gate
     │
     ▼
CV-1.5  Domain router (framed or wide-field?)
     │                    │
     ▼                    ▼
[pre-framed]        CV-2  Detection
dermoscopic              │
or clinical         0 candidates → STOP
     │              1+ candidates → crop+normalize
     └──────────────────►
                         ▼
                    CV-3  Segmentation
                         │
                         ▼
                    CV-4  Classification
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     CV-5 Explain   CV-6 Uncertainty  CV-7 Temporal
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                    CV-8  Risk engine
                         │
                         ▼
              Structured clinical/risk context
```

**Key architectural decisions (all committed in docs/):**
- CV-1.5 routes on FRAMING (pre-localized vs wide-field), NOT on
  modality (dermoscopic vs smartphone). PAD-UFES-20 is lesion-centric
  (visually confirmed), so it skips CV-2 and goes straight to CV-3.
- CV-8 receives CV-5 + CV-6 + CV-7 as CONVERGENT inputs, not a relay.
  Uncertainty + temporal + classification evidence ALL feed the risk
  engine — absence of temporal data must not auto-produce a low-risk
  output (Case D reasoning).

---

## Dataset roles (locked, from CV_DATASET_SPEC_v1.0)

| Dataset | Role |
|---|---|
| ISIC 2019 (= HAM10000 + BCN20000 + MSK) | CV-4 classification training |
| PAD-UFES-20 | Smartphone/clinical domain transfer validation for CV-4 |
| ISIC 2018 Task 1 | CV-3 segmentation training (512×512, binary masks) |
| iToBoS 2024 | CV-2 detection training (wide-field TBP, bboxes, zero-lesion images) |
| UQ Longitudinal | CV-7 temporal "What Changed?" feature |
| SCIN | Out of scope for CV pipeline; reserved for future symptom/diversity work |
| PH2 | Small CV-3/CV-4 auxiliary/eval source |

**Critical: NEVER use HAM10000 separately alongside ISIC 2019** — HAM is
a subset of ISIC 2019. Using both double-counts.

**iToBoS/UQ/SLICE-3D overlap:** documented, unresolved at participant level.
Do NOT treat iToBoS and UQ Longitudinal as jointly independent for a
single reported evaluation claim until resolved.

**iToBoS license: CC BY-SA 4.0** — permits commercial use but carries
ShareAlike. Legal review required before any CV-2 production/commercial
distribution. Tracked in docs/cv2_detection_spec.md Section 13.

---

## CV component status

### CV-1 — Image Quality Gate
**Status: RECALIBRATED (2026-09-01).** Real-image rejection on PAD-UFES
was **13.6% → 1.42%** after recalibration; genuine severe synthetic
degradation still caught ≥95% (all types). Full result:
`analysis/quality/cv1_recalibration/result.md`, spec:
`docs/cv1_recalibration_spec.md`.

The prior "validated on synthetic degradation only" caveat turned out to
hide a real defect, not just an untested gap: `resolution` and `blur`
both derived from Laplacian variance (r=0.58), double-penalizing one
soft image as two issues, and thresholds calibrated on synthetic data
did not transfer to real clinical images. Measured impact before the
fix: no CV-1 signal predicted CV-4 success (resolution r=−0.131,
p=0.014 — significant but inverted; others n.s.), and CV-4 was actually
MORE accurate on CV-1-rejected images than accepted ones (85.4% vs
67.8%, p=0.018 after controlling for class mix).

Fix: `resolution_signal` now measures dimensions only (`src/quality/signals.py`);
`assess_image` uses a two-tier design — advisory issues feed capture
guidance without blocking, a separate `unusable_*` tier blocks on
genuine unusability (`src/quality/assessment.py`). One threshold
(`unusable_contrast`) was adjusted post-hoc to close a gap the first
pass left open; both real-rejection and severe-degradation criteria
hold simultaneously. Downstream regression check: 100% agreement with
prior CV-4 predictions on the images already being assessed.

Quality assessment and guidance: `src/quality/`. Robustness harness:
`analysis/quality/cv1_robustness/` (re-run against the new thresholds).

### CV-1.5 — Domain Router
**Status: COMPLETE.** See "Completed: CV-1.5 Domain Router" below for
the full result. ResNet18 classifier, 1.000/1.000 on held-out set.
`src/routing/classifier.py::route_image`. PAD-UFES-20 is pre-framed
(visually confirmed), so it takes the pre-framed path directly to CV-3.
Wired into the assembled pipeline
(`src/inference/orchestrator.py`); `src/inference/pipeline.py` remains
the separate CV-4-only path.

### CV-2 — Lesion Candidate Detection
**Status: BASELINE ACCEPTED AT FLOOR — refinement deferred**
See `docs/cv2_status.md` for full detail. Summary:

**Model:** YOLO11n (nano) trained on iToBoS 2024.
**Baseline checkpoint:** `runs/cv2/b1_1280/weights/best.pt` (network volume).
Note: E (YOLO11s) is equally valid but nano preferred (3.6x smaller,
same performance).

**Revised acceptance metrics (Section 22, `docs/cv2_section22_finalized.md`):**
- Image-level recall (primary gate): B1 = 0.8098, target ≥ 0.90 → AT FLOOR
- Zero-lesion FP burden median (gate): 0.00, target ≤ 1 → PASSES
- Zero-lesion FP burden p90 (gate): 1.00, target ≤ 2 → PASSES

**Experiments run (all documented in `analysis/quality/cv2_pathological_audit/`):**
1. Threshold sweep → no usable operating point.
2. B0→B1 resolution (1024→1280) → no effect on locked metrics.
3. D — sun-damage hard-negative oversampling (8x) → no effect.
4. E — YOLO11s capacity → no effect (image-level recall +0.0009).

**Deferred refinement:** tiling/SAHI — **effectively CLOSED** (gate
re-scoped 2026-09-01). The original gate asked for an end-to-end risk
measurement iToBoS cannot support (no diagnosis labels), and never
needed to: a CV-2 miss costs structurally (the lesion is invisible
downstream), which was already measured. Wide-field is not the primary
input path, and an incidental user wide frame is a phone photo rather
than TBP imagery — so tiling would tune against a domain the product may
never see. The operative CV-2 question is now the iToBoS→phone domain
gap. Full reasoning in `docs/cv2_status.md`.

**Constraint on downstream development:** do NOT develop CV-3 on CV-2's
real detection crops. CV-3 develops on ISIC ground-truth crops. The
CV-2→CV-3 interface was validated separately — geometry robust, domain
measured (see the completed sections below).

### CV-3 — Lesion Segmentation
**Status: BASELINE COMPLETE**
- U-Net from scratch, `src/segmentation/model.py` (`build_model()`).
- Trained on ISIC 2018 Task 1 (512×512, binary masks, BCE+Dice loss).
- **Baseline: Dice 0.8640, IoU 0.7851** (frozen test set).
- Checkpoint: `checkpoints/cv3_512/best.pt` (locally and on volume).
- Input: 512×512, BGR→RGB, /255, CHW. Squash resize (NO aspect ratio
  preservation). Loads via:
  `build_model()` + `torch.load(...)["model_state_dict"]`.
- Domain limitation MEASURED (2026-09-01): on real CV-2 detection crops
  from iToBoS TBP, 78% of masks rated reasonable — ~22% fail, dominated
  by fragmentation (`analysis/quality/cv3_domain_itobos/result.md`).
  Note the assembled-pipeline run saw only 1.0% degenerate masks on
  wide-field candidates; different crop populations, both recorded.

### CV-4 — Classification (Native Diagnosis)
**Status: SUBSTANTIALLY ESTABLISHED**
- ResNet-50, ISIC 2019 (8 classes: AK/BCC/BKL/DF/MEL/NV/SCC/VASC).
- Selected model: weighted ResNet-50 (C1 partial fine-tune for PAD-UFES).
- **ISIC test: Macro-F1 0.5756, MEL recall 56.6%.**
- **PAD-UFES transfer (C1): Macro-F1 0.5996.**
- Key finding: classifier outputs NATIVE DIAGNOSIS probabilities. The
  risk/action category (Low/Suspicious/High) lives in the RISK ENGINE
  (CV-8), NOT in the classifier. This is the correct architecture.
- SCC/BCC confusion documented (`analysis/scc_bcc/`): representation
  overlap confirmed, not fixable by reweighting/SupCon alone. Accepted
  as a known limitation.
- Safety bottleneck (Phase 4): 32 Tier-1 errors (high-risk→non-high-risk)
  dominated by BCC→ACK. Documented in `analysis/product_eval/`.
- **Out-of-domain behavior INVESTIGATED (2026-09-01).** The 32.7% URGENT
  rate / 12.9% high-risk-per-candidate rate on wide-field TBP crops
  (`analysis/product_eval/cv1_cv4_assembly/result.md`) is NOT a general
  miscalibration — confidence in/out of domain is similar overall
  (0.733 vs 0.760). It IS concentrated in BCC and ACK specifically
  (confidence 0.578/0.590 out-of-domain vs 0.743/0.773 in-domain),
  consistent with — not a new instance separate from — the already-
  documented SCC/BCC embedding overlap above. Visual audit + a new
  crop-level contrast signal confirm why: BCC/ACK predictions are
  disproportionately diffuse, low-contrast, hair-dominated crops (BCC
  52.1% / ACK 88.1% below contrast 0.20 at n=3336, vs MEL 2.4%), not
  coherent lesion shapes. **No retraining attempted** (SCC/BCC question
  already closed) and **no filtering added** (would recreate the
  silent-miss failure mode). Fix: `crop_blur`/`crop_contrast` evidence
  fields added to every candidate (`src/inference/orchestrator.py`,
  reusing `src/quality/signals.py`) — disclosure, not a gate, matching
  the CV-3-mask precedent. Full result:
  `analysis/product_eval/cv4_domain_evidence/result.md`, spec:
  `docs/cv4_domain_evidence_spec.md`.
- Checkpoints: `checkpoints/isic2019_resnet50_weighted_best.pt` (ISIC
  baseline), `checkpoints/pad_ufes_c1_partial_finetune_best.pt` (transfer).

### CV-5 — Explainability
**Status: COMPLETE (v1) — 2026-09-01.**
`analysis/product_eval/cv5_explainability/result.md`, spec:
`docs/cv5_explainability_spec.md`. **How this was built and how it fits
CV-3/CV-4/CV-6 architecturally: `docs/cv5_cv6_evidence_architecture.md`**
(has the full pipeline ASCII diagram). Mask-contour overlay (reused from
`scripts/validate_cv3_domain_itobos.py::draw_overlay`) + Grad-CAM
heatmap (`src/explainability/gradcam.py`, bypassing
`NativePredictor.predict()`'s `@torch.no_grad()` to reach the model
directly). Required one additive method,
`SharedResNetXXBackbone.forward_conv_features()`
(`src/models/native_classifier.py`) — verified to exactly reconstruct
the existing `forward()` output via `F.adaptive_avg_pool2d`, and all
existing CV-4 regression tests (7) pass unmodified.

New cross-check evidence: `gradcam_mask_iou` — does CV-4's attention
overlap with CV-3's independently-computed mask? First time these two
components have been compared to each other at all. Visual audit (3
crops, MEL/BCC/NEV) confirms the heatmap lands on the lesion, not
background/hair, in every case.

**Deliberately not wired into `DermaSensePipeline.predict()`'s hot
path** — `explain_candidate()` is invoked on demand (needs a backward
pass, costlier than CV-3/CV-6's forward-only evidence; produces overlay
images, which don't fit `CandidateResult`'s scalar-evidence CSV
pattern). Same evidence-not-decision principle as CV-3/CV-6 — no
policy built around `gradcam_mask_iou`, reserved for CV-8.

### CV-6 — Uncertainty
**Status: COMPLETE (v1, evidence layer) — 2026-09-01.**
`analysis/product_eval/cv6_uncertainty/result.md`, spec:
`docs/cv6_uncertainty_spec.md`. Same cross-cutting doc as CV-5:
`docs/cv5_cv6_evidence_architecture.md`. Three zero-new-training evidence signals
on every candidate: ensemble disagreement (seed42 vs seed123, opt-in via
`--ensemble`, off by default — doubles CV-4 inference cost),
temperature-calibrated confidence (T=1.25, fit on PAD-UFES val, ECE
0.0596→0.0401; post-hoc on probabilities only, no change to
`NativePredictor`), and `crop_contrast`/`crop_blur` (already existed,
now documented as CV-6 evidence). **Does NOT modify
`src/risk/safety_gate.py`** — evidence only, per the same
dependency-direction principle as the CV-3-mask and `crop_contrast`
decisions.

Pre-committed corroboration check against the CV-4 domain-evidence
finding (does ensemble disagreement independently spike on BCC/ACK, the
classes `crop_contrast` flagged?) came back **partial, and the
exceptions are the useful part**: BCC replicates (moderate disagreement,
consistent with the already-documented SCC/BCC overlap), but ACK does
NOT (both ensemble members confidently agree on ACK's low-information
crops — a "confidently wrong together" failure mode `crop_contrast`
alone catches but disagreement can't), and MEL shows the HIGHEST
disagreement of any class despite being visually coherent (a genuine
MEL/NEV boundary-ambiguity signal, present in-domain as much as out,
that `crop_contrast` alone would never catch). The two signals are
complementary, not redundant — direct evidence for why CV-8 needs CV-5/6/7
as convergent inputs rather than one collapsed score.

### CV-7 — Temporal ("What Changed?")
**Status: STAGE 1 COMPLETE AND EVALUATED (2026-09-02) — ruler
calibration, per-visit measurement, delta/verdict assignment, the
assembled `TemporalPipeline`, and the clinical-signal-validity
evaluation are all done. Decision: Stage 1 (classical, deterministic —
no learned model) is sufficient. No Stage 2 and therefore no RunPod
GPU training is needed for CV-7. Wiring into CV-8 and real pairing
logic for a user's upload history is the remaining integration work.**
`docs/cv7_temporal_rag_integration_spec.md` — the product/architecture
spec, decided by the user, covers two things:

1. **What CV-7 computes**: a structured, numeric temporal verdict per
   lesion pair (`STABLE | GROWING | SHRINKING | CHANGED_COLOR |
   NO_PRIOR_DATA`, magnitude, per-feature deltas for size/border/color,
   confidence), consumed by CV-8 alongside CV-4's diagnosis and CV-6's
   uncertainty — same convergence pattern as the rest of the pipeline.
2. **The CV↔RAG chatbot boundary** (the user is separately building a
   RAG-based explainable-AI chatbot alongside this CV pipeline): the
   chatbot is strictly downstream, reading CV-8's structured JSON output
   to narrate it in natural language — it never generates the temporal-
   change or risk claim itself. Rejected alternative: letting an LLM
   narrate "what changed" directly from images, which would make a
   safety-relevant claim generated rather than computed/verified — same
   dependency-direction principle used for CV-3's mask and every CV-5/6
   evidence signal (`docs/cv5_cv6_evidence_architecture.md`). The
   handoff contract (JSON schema) is locked in the spec; two
   discrepancies against the current codebase (`risk_category` vs.
   `ProductAction`; PAD-UFES vs. ISIC2019 class taxonomy) are flagged
   there for CV-8 to reconcile, not resolved yet.

**Data acquisition (blocker 2) — bounded sample staged (2026-09-02),
full dataset deferred.** Full result:
`analysis/quality/cv7_temporal_data/result.md`. Downloaded UQ
Longitudinal (62GB zip) locally, inspected without full extraction
(`zipfile`/`unzip -l`): the Dermoscopic Images folder (63.77GB, 35,914
files) essentially *is* the documented longitudinal subset — unlike
iToBoS, there was no large non-longitudinal portion to trim away.
Filtered to true longitudinal (≥2-visit) participants: 331 / 57.7GB /
7,672 lesions, closely matching the paper's 340/7,038 figures.

Two independent storage constraints (local ~23GB free, RunPod volume
~25-30GB free) made the full 58GB set unstageable immediately, so —
same discipline as CV-1.5's 150-image held-out set and CV-3's 1,000-crop
sample — staged a **bounded, seeded, stratified sample first**: 30
participants (16 General, 14 HighRisk, seed 42), 2,772 images + 3
metadata files, 5.12GB. Uploaded to
`s3://4tlwcuo1xg/dermasense/data/raw/uq_longitudinal/` and verified
byte-for-byte against the local extraction (2,775 objects, 5,116,534,902
bytes both sides). Local copy also kept at `data/raw/uq_longitudinal/`
(gitignored); source zip retained at `data/raw/UQ_zip/` in case more
participants are sampled later.

`src/temporal/__init__.py` still empty. History of how the original two
blockers were investigated: `docs/cv7_temporal_blockers.md`.

**Technical spec written (2026-09-02):** `docs/cv7_temporal_technical_spec.md`,
after actually inspecting the staged sample's images and metadata:

- **Real ground truth exists**: a `Diagnosis` field per lesion (96.5%
  benign dataset-wide; melanoma/BCC/SCC/AK/nevus/etc. otherwise),
  constant across all of that lesion's visits — a final outcome label,
  not a per-visit-in-time one (verified: 0/9,382 lesions have
  conflicting diagnoses across visits). This makes CV-7's measured
  change and the diagnosis label independent, testable signals.
- **A physical mm ruler is baked into every image frame**, and multiple
  cameras were used (Canon T6i 82%, Veos SLR 17%, minor others) — so
  real-world-unit size measurement is possible via classical image
  processing, calibrated **per image**, not a fixed constant.
- **Design: Stage 1 is classical/deterministic, no training** — reuses
  CV-3 directly for segmentation (this is real dermoscopic imagery,
  CV-3's actual training domain, unlike iToBoS), computes size/color/
  border deltas from ruler-calibrated measurements, assigns the
  verdict via calibrated thresholds. Mirrors CV-1.5's Stage-1-before-
  training discipline exactly. A learned Stage 2 is explicitly deferred
  unless Stage 1 proves specifically insufficient.
- **Malignant-enrichment pull DONE (2026-09-02).** The 30-participant
  sample only had 4 malignant-outcome lesions — not enough for clinical
  validation. Pulled the 54 new participants (of 57 dataset-wide, 3
  already in the sample) carrying a longitudinal malignant lesion
  instead of resizing: 10.96GB, 5,979 files. **Combined total verified
  byte-for-byte against the RunPod bucket: 8,754 objects,
  16,081,485,217 bytes** — both batches' file counts and sizes sum
  exactly. Staged dataset now has **99 malignant-outcome lesions with
  visit pairs** (up from 4). Full record:
  `analysis/quality/cv7_temporal_data/result.md`.
- **No resize was needed.** Local disk got tight mid-extraction (97%
  used, 7GB free) — both extracted batches were deleted after
  byte-for-byte bucket verification (re-extractable from the retained
  source zip any time), restoring ~25GB free locally. RunPod volume
  likewise took the full 16.08GB without a resize. A full resize
  (~70-80GB, ~$5-6/month, for the remaining ~274 participants) stays
  deferred until a learned Stage 2 is actually justified.

**Ruler calibration implemented and gated (2026-09-02).**
`src/temporal/calibration.py`, full result:
`analysis/quality/cv7_temporal_data/calibration_result.md`. Detects the
per-image mm ruler via a probabilistic Hough transform (two other
approaches — darkness/shape blob heuristics, then the same with HSV
saturation filtering — were tried and rejected first, each on real
images, not assumed). **Measured confident-calibration rate: 4.0%** on
a 200-image random sample — low, but the confident results are tight
(262.2–269.0 px/mm) and match the manufacturer-corroborated ~265px/mm
almost exactly, confirming the gap is detection sensitivity (mostly
"ruler not found at all" — hair, lesion-curve occlusion, non-fixed
handheld framing), not a wrong assumption. **Decision: ship the gate as
designed** — a failed calibration returns `NO_PRIOR_DATA` for size only
(color/border deltas are unaffected); improving detection sensitivity
is a separately-scoped follow-up, not attempted now.

**Measurement implemented (2026-09-02).** `src/temporal/measurement.py`,
full result: `analysis/quality/cv7_temporal_data/measurement_result.md`.
Before writing it, verified CV-3 actually fits this domain: a 100-image
random sample (seed=7) measured **5.0% degenerate-empty masks, 0%
degenerate-full**, far better than the ~22% fragmentation rate on
iToBoS TBP crops, confirming the technical spec's expectation. Reuses
CV-3 directly (no retraining); `measure_lesion()` mirrors calibration's
fail-loud design with two independent gates — `valid` (was a lesion
mask found at all) and calibration-confidence (whether `diameter_mm`/
`area_mm2` are real-unit; pixel-space `area_fraction` and border
`compactness` are scale-invariant and always available when valid). A
real multi-blob case found during validation (two separate lesions in
one frame) is resolved by keeping only the largest connected
component, since the dataset names one lesion per image.

**Delta/verdict implemented, thresholds calibrated (2026-09-02).**
`src/temporal/delta.py`, full result:
`analysis/quality/cv7_temporal_data/delta_calibration_result.md`.
Produces the locked verdict (`STABLE | GROWING | SHRINKING |
CHANGED_COLOR | NO_PRIOR_DATA`) plus per-feature size/border/color
deltas and a confidence, matching
`docs/cv7_temporal_rag_integration_spec.md`'s JSON contract exactly.
`scripts/calibrate_cv7_thresholds.py` measured real deltas on a
300-pair bounded sample (seed=11): border/color thresholds were set
from real percentiles (280 pairs with valid masks). **The size/growth
threshold could not be calibrated the same way** — only 1/300 pairs
(0.3%) had confident ruler calibration on both visits, the expected
compounding result of calibration's 4.0% single-image rate (0.04²≈
0.16%). Rather than process most of the 8,751-image corpus chasing
double-confident pairs for a feature that stays structurally rare
regardless, the growth threshold ships as an explicitly-flagged
provisional placeholder — not silently treated as equally well-founded
as border/color. Also found and documented as a known limitation:
color-delta is confounded by cross-visit lighting/camera variation
(median observed delta already exceeded a naive threshold), so the
calibrated color threshold is set conservatively high. Descriptive-only
secondary finding (n=19, not validated): malignant-outcome lesions
showed a notably larger mean border-shape delta (3.08) than benign
(0.89) in this sample.

**Assembled into a pipeline (2026-09-02).** `src/temporal/pipeline.py::TemporalPipeline`
wires `calibration.py` → `measurement.py` → `delta.py` into one entry
point, `assess_pair(earlier_image, later_image)`, mirroring
`src/inference/orchestrator.py`'s shape (`from_checkpoint` loads CV-3
once; one validated entry point; an immutable result with `to_dict()`).
Kept as a separate class rather than folded into `DermaSensePipeline`,
since CV-7 takes a PAIR of same-lesion images, a different contract
shape from CV-1→CV-4's one-image input. `TemporalResult.to_dict()`
matches the locked JSON contract's `temporal` sub-object exactly
(`verdict`, `magnitude`, `confidence`, `per_feature_deltas`,
`compared_timestamps`), with one flagged deviation: `per_feature_deltas.size`
serializes as `None` (not `0.0`) when either visit lacks confident
calibration, since a `0.0` there would misrepresent "not measured" as
"no change" — the exact failure mode this project's fail-loud pattern
exists to prevent. Full suite: 123/123 passing.

CV-7's module chain is now complete end-to-end. Remaining work is
integration, not CV-7 itself: pairing logic for a real user's upload
history (this module takes two already-selected images), wiring into
CV-8, and reconciling the two discrepancies already flagged in
`docs/cv7_temporal_rag_integration_spec.md`.

**Stage 1 clinical-signal evaluation DONE — no training needed
(2026-09-02).** `scripts/evaluate_cv7_stage1.py`, full result:
`analysis/quality/cv7_temporal_data/stage1_evaluation_result.md`. This
answers the question left open when the technical spec was written
("does measured change actually correlate with malignant outcome?"),
now that the malignant-enrichment data is staged. Pre-registered before
running: Fisher's exact test on non-STABLE verdict rate + Mann-Whitney
U on delta magnitude, p<0.05 fixed in advance, run over ALL 101
malignant-outcome lesion pairs vs. a bounded random 300-pair benign
sample (seed=17). **Both tests came back significant in the expected
direction**: malignant pairs non-STABLE at 25.8% vs. 13.7% benign
(p=0.0135); magnitude distribution significantly higher (p=8.2×10⁻⁸).

**Decision: Stage 1 (the classical pipeline — no learned model, no
training) is sufficient. No Stage 2 and therefore no RunPod GPU
training is needed for CV-7**, per the technical spec's own
anti-rabbit-hole boundary (Stage 2 only if Stage 1 shows a genuine,
specific insufficiency — this evaluation didn't find one). Caveats
documented in the result doc: n=89 scored malignant pairs, one visit
pair per lesion, and a plausible photography-condition confound (more
closely monitored/biopsied lesions may simply be photographed under
more variable conditions) not distinguished from real biological
change — the result is significant, not a definitive clinical claim.
CV-7 proceeds to CV-8 integration as a classical component.

### CV-8 — Risk Engine / Severity
**Status: CV-4+CV-5+CV-6+CV-7 ALL WIRED INTO THE PIPELINE (2026-09-02).
Every CV phase now feeds `DermaSensePipeline.predict()` — see "Baseline
v1" below.**
- `src/risk/action_mapping.py` and `src/risk/safety_gate.py`: unchanged,
  still the internal `ProductAction`/gate logic.
- `src/risk/convergence.py::assess_risk()` — NEW. Converges a
  `CandidateResult` (CV-4 diagnosis + CV-6 calibrated confidence) and
  an optional `TemporalResult` (CV-7) into the exact JSON contract
  locked in `docs/cv7_temporal_rag_integration_spec.md`
  (`lesion_id/diagnosis/risk_category/risk_reason/temporal/uncertainty/quality_flags`).
  Resolves discrepancy 1 from that spec: `risk_category`
  (LOW/MEDIUM/HIGH) is a new field derived from `ProductAction`
  (`URGENT_EVALUATION→HIGH, EVALUATE_SOON→MEDIUM, MONITOR→LOW,
  UNKNOWN→HIGH` fail-safe), not a replacement for it. Discrepancy 2
  (ISIC 8-class vs. PAD-UFES 6-class `native_class` taxonomy) stays
  open, deliberately — orthogonal to this work, passes through
  whatever `CandidateResult.predicted_class` actually is.
  **The one real design decision**: CV-7's verdict can escalate
  `risk_category` by one step (never de-escalate) — only for
  `GROWING`/`CHANGED_COLOR`, only when `magnitude >= 1.0` and
  `confidence >= 2/3`. `SHRINKING`/`STABLE`/`NO_PRIOR_DATA` never
  affect risk in either direction, since an absent signal is equally
  consistent with "not changing" and "couldn't measure" (calibration's
  4% size-coverage, measurement's 5% mask-miss rate). Escalation also
  forces `requires_review=True` regardless of CV-4/CV-6's own
  decision. This rule acts on a signal Stage 1's own evaluation already
  found predictive (`stage1_evaluation_result.md`: p=0.0135 non-STABLE
  rate, p=8.2e-8 magnitude), not an untested assumption. 15 new tests,
  full suite 138/138 passing.
- **Wired into `DermaSensePipeline.predict()` (2026-09-02).**
  `predict()` gained `lesion_id`, `prior_image_bgr`, `prior_timestamp`,
  `current_timestamp` params. `TemporalPipeline` is now constructed
  once in `__init__`, reusing the SAME CV-3 segmenter already loaded
  (no extra checkpoint, no extra memory). `_run_candidate` now always
  calls `assess_risk()` and attaches the result as
  `CandidateResult.risk_assessment` — populated for every candidate,
  with or without a prior image (CV-8 degrades to `temporal=None`
  gracefully).

  **Pairing rule** (`_resolve_temporal_pairing`, a pure/unit-testable
  function): temporal comparison only runs when the current image has
  exactly one candidate. With more than one, which detected lesion a
  supplied prior image corresponds to is genuinely ambiguous — this
  pipeline has no cross-image lesion re-identification and was never
  asked to build one, so it skips and flags
  (`PRIOR_IMAGE_PAIRING_AMBIGUOUS`) rather than guessing, the same
  fail-loud discipline as calibration.py's confidence gate.
  `lesion_id` resolution (`_candidate_lesion_id`) follows the same
  logic: a caller-supplied id applies directly only when unambiguous
  (one candidate); otherwise each candidate gets its own suffixed id.

  **Verified end-to-end on real checkpoints**: two real (clinically
  unrelated, since no true visit-pair images were used for this
  smoke test) PAD-UFES images produced a genuine `CHANGED_COLOR`
  verdict (confidence 2/3, border+color only — no ruler in these
  photos, exactly as expected for real product images with no
  physical scale reference) that escalated `EVALUATE_SOON` to `HIGH`
  and forced `requires_review=True`
  (`tests/test_pipeline_assembly.py::test_prior_image_wires_real_cv7_comparison`).
  9 new pure unit tests + 2 new integration tests, full suite
  147/147 passing.

  **Finding worth noting**: this confirms CV-7's border/color signal
  (already shown predictive in `stage1_evaluation_result.md`) works
  through the full pipeline even with ZERO ruler coverage — the
  realistic case for actual DermaSense users, who will essentially
  never have a physical mm ruler in their photos. Size-based escalation
  will be correspondingly rare in production, by design, not a bug.

- **`quality_flags` now surfaces CV-1/CV-3/CV-6 evidence too
  (2026-09-02).** `assess_risk()` previously only used CV-4's diagnosis
  and CV-6's *calibrated confidence* — the mask/crop/ensemble evidence
  fields already computed on `CandidateResult` never reached the JSON
  contract. Now they do, as disclosure-only flags (never affect
  `risk_category` or `requires_review` — verified by a dedicated test):
  `DEGENERATE_MASK`/`MASK_TOUCHES_BORDER` (CV-3, direct booleans),
  `LOW_CROP_CONTRAST` (`crop_contrast < 0.20`, the cutoff independently
  validated in `docs/cv4_domain_evidence_spec.md` — BCC/ACK crops fall
  below it 58.7%/similarly vs. MEL's 5.1%), `LOW_CROP_BLUR`
  (`crop_blur < 0.15`, borrowed from CV-1's whole-image advisory
  threshold — honestly flagged as weaker evidence than
  `LOW_CROP_CONTRAST` since blur was never independently validated at
  crop scale), `ENSEMBLE_DISAGREEMENT` (`ensemble_agree is False`).
  `ensemble_probability_distance`/`ensemble_confidence_spread` got no
  flag — no calibrated cutoff exists for either anywhere in this
  project, so none was invented. 10 new tests, full suite 157/157.
- **CV-5 wired (2026-09-02).** `CandidateResult.gradcam_mask_iou` — new
  field, opt-in via `compute_gradcam=True` on `__init__`/
  `from_checkpoints` (off by default: Grad-CAM needs a real backward
  pass through the classifier, `torch.enable_grad()`, a materially
  different cost from CV-6's forward-only ensemble evidence). Recorded
  raw, like `ensemble_probability_distance`/`ensemble_confidence_spread`
  — no calibrated threshold exists for `gradcam_mask_iou` anywhere in
  this project (`docs/cv5_cv6_evidence_architecture.md` always framed
  it as a raw cross-check), so no flag was fabricated for it. Verified
  end-to-end on a real checkpoint (IoU 0.187 on a real PAD-UFES image).
  2 new tests, full suite 159/159.
- **Not yet done**: the lesion-history store itself (which prior image
  to supply for a given lesion_id) — a product/backend persistence
  concern, out of this pipeline's scope by design, not attempted here.

### Baseline v1 — full CV-1→CV-8 pipeline (2026-09-02)

Every CV phase built this session now feeds one call:
`DermaSensePipeline.predict(image_bgr, *, lesion_id=None,
prior_image_bgr=None, prior_timestamp=None, current_timestamp=None)`.

```
image -> CV-1 quality gate -> CV-1.5 router -> [CV-2 if wide_field]
      -> per candidate: CV-3 mask -> CV-4 diagnosis -> CV-5 Grad-CAM (opt-in)
                       -> CV-6 calibration/ensemble (opt-in)
                       -> CV-7 temporal (if prior image + unambiguous)
                       -> CV-8 assess_risk() -> risk_assessment
```

Every candidate always gets a `CandidateResult.risk_assessment`
(the locked JSON contract: `lesion_id/diagnosis/risk_category/
risk_reason/temporal/uncertainty/quality_flags`) — CV-8 degrades
gracefully at every optional input (no prior image, no ensemble, no
Grad-CAM), so "we didn't have X" is always a flag or a `None`, never a
silently wrong value. This is the same fail-loud principle applied
consistently since CV-1: `PipelineOutcome` for whole-image
non-assessment, `RulerCalibration.confident` for per-image calibration,
now `risk_assessment`'s optional fields for per-signal availability.

**What "baseline v1" deliberately does NOT include, by design, not
oversight:**
- The lesion-history store (which prior image to supply) — a
  product/backend persistence concern this pipeline was never asked to
  own; the caller supplies `prior_image_bgr` if it has one.
- Cross-image lesion re-identification for wide-field multi-candidate
  images — genuinely ambiguous without it, so CV-7 pairing is skipped
  and flagged (`PRIOR_IMAGE_PAIRING_AMBIGUOUS`) rather than guessed.
- The two flagged discrepancies against the RAG contract: `native_class`
  taxonomy (PAD-UFES 6-class vs. the contract's ISIC 8-class example)
  and CV-2's known tiling limitation — both pre-existing, deliberately
  out of this scope, documented where they were found.
- Any Stage 2 (learned) model for CV-1.5 or CV-7 — both stayed
  Stage-1-only after their own evaluations found no need to escalate.
- CV-6/CV-5 are opt-in (cost tradeoffs — extra forward passes / a
  backward pass), off by default; a caller wanting them passes
  `additional_ensemble_checkpoints=...` / `compute_gradcam=True`.

**What this milestone means**: every CV phase has at least one path
from raw image to the final JSON contract, verified end-to-end on real
checkpoints and real images (not just unit-tested in isolation). It
does not mean every signal fires on every image — most won't (no
ruler, no ensemble/Grad-CAM by default, no prior image) — by design,
per each component's own documented, measured coverage.

**Delivered to the RAG collaborator (2026-09-02):**
`docs/cv8_sample_outputs/sample_outputs.json` — 5 real
`RiskAssessment.to_dict()` payloads (real checkpoints, real images: 4
from the staged UQ Longitudinal sample, 1 from PAD-UFES), covering
first-visit/no-history, a stable returning visit, a real CV-7
escalation (`NEV`→`MONITOR` base pushed to `MEDIUM` by `CHANGED_COLOR`),
a "prior image supplied but unmeasurable" case, and a disclosed
quality flag alongside a normal result — picked to show the range of
real shapes, not a curated narrative. `docs/cv8_sample_outputs/README.md`
documents the schema and calls out the two things a parser MUST
handle: `per_feature_deltas.size` is frequently `null` (not `0.0` —
real users have no ruler, so this is the common case, not an edge
case), and `native_class` is currently PAD-UFES's 6-class taxonomy,
not the contract's own 8-class ISIC example (the still-open
discrepancy). Regenerable via
`python -m scripts.generate_cv8_sample_outputs`. This is a static
example set, not a live delivery mechanism — no API/queue/file-drop
integration was built or decided; that remains open.

**Planning doc: `docs/build_on_baseline_1.md` (2026-09-02).** Covers
the three open questions after baseline v1, each trigger-gated per
this project's anti-rabbit-hole discipline: (A) the live-feed delivery
plan for the RAG collaborator (prerequisite unknowns, a recommended
minimal sync-HTTP default, explicitly what NOT to build yet); (B) a
14-item, cited weakness audit across CV-1→CV-8, each with an issue,
evidence, a concrete trigger condition, and a bounded first step — no
item without a stated trigger; (C) the CV-7 full-dataset training
question, concluding it is **not currently triggered** — Stage 1 was
already shown sufficient — with an explicit recommendation AGAINST a
full-dataset pull just to calibrate the near-useless (real users have
no ruler) size threshold, and a pointer-level resource plan for a
learned Stage 2 held in reserve until one of three named trigger
conditions is actually met.

**RunPod volume snapshot taken before termination (2026-09-02).**
`analysis/quality/runpod_volume_snapshot/result.md`. User's plan
(terminate the volume now, re-provision only what's needed when a
Phase 2 trigger fires) was sound, but the paper-trail snapshot found
the volume was a full repo mirror, not just the 5 research datasets as
initially assumed — including `checkpoints/`, `runs/`, and
`evaluation/`. Diffing against local found **8 trained checkpoint
files that existed only on the volume** (`cv3_768/*` entirely, plus
5 PAD-UFES variant checkpoints including the 107MB SupCon one) —
these would have been unrecoverable on termination. **Downloaded and
MD5-verified against S3's own ETag before giving the go-ahead**, along
with 106MB of `runs/` and 86.5MB of `evaluation/` logs also missing
locally. Verdict: safe to terminate now — everything else on the
volume (the 5 datasets, `.venv`, `.cache/pip`, `.git`, source/docs) is
either re-obtainable from source or redundant with GitHub.

---

## Repo structure conventions

```
src/<component>/        source code by CV phase
scripts/                flat, all scripts at top level, prefix = component
                        (e.g. evaluate_cv2.py, train_cv3.py)
analysis/<topic>/       investigation outputs (CSVs, summaries)
  analysis/quality/     dataset/pipeline audits
  analysis/product_eval/ end-to-end product metrics
  analysis/scc_bcc/    CV-4 SCC/BCC deep-dive (complete, closed)
docs/                   specs and decision records
checkpoints/            model checkpoints (GITIGNORED -- on volume/local)
data/                   raw + splits (large files gitignored)
  data/raw/             SYMLINK (2026-09-05) -> /mnt/hdd/dermasense_data/raw/
                        All 5 datasets (isic2018, isic2019, itobos,
                        pad_ufes, UQ_zip -- 127.4GB, 58,264 files) moved
                        to the external HDD to free local disk space.
                        Copied via rsync, verified byte-for-byte
                        identical (file count + total bytes matched
                        exactly) before the local copy was deleted and
                        replaced with this symlink. Every existing
                        script/path reference (`data/raw/...`, relative
                        to repo root) works unchanged -- verified via a
                        real cv2.imread() through the symlink and the
                        full test suite (159/159 passing). If
                        /mnt/hdd is ever unmounted or unavailable,
                        every `data/raw/*` path will fail to resolve --
                        check the symlink target exists before
                        debuging a "file not found" as a code bug.
tests/                  central test suite (CV-3 tests; CV-2 detection
                        tests are inside src/detection/ -- inconsistency,
                        flagged for future cleanup)
```

**Key conventions:**
- Scripts use `python -m scripts.<name>` (not bare `python scripts/`) for
  any script that imports from `src/`.
- Checkpoint loading: `build_model()` + `torch.load(path)["model_state_dict"]`
  for CV-3. CV-4 uses similar pattern (see `scripts/evaluate_checkpoint.py`).
- Never `pip install` with venv activated — always
  `/path/to/python -m pip install` to ensure correct interpreter.
- Never push from a pod without verifying the remote hasn't moved first
  (`git fetch origin` + `git log --oneline origin/main -3`).

---

## Completed: CV-2 → CV-3 Interface Validation (geometry)

**Result: ROBUST.** `analysis/quality/cv2_cv3_interface/validation_run.txt`
(2026-08-31). Harness self-check passed (margin=1.0/offset=0.0 Dice
0.8631, recovers ~baseline 0.8640). Realistic crop (margin=0.25/offset=0.1)
Dice 0.8742 ≥ 0.75 gate. Worst cell in the whole 15-cell grid
(margin=0.0/offset=0.2) still 0.8160 — no collapse anywhere swept.
`src/inference/crop_normalize.py`'s existing `margin=0.25` default is
confirmed as the (near-)best-performing value in the grid — no code
change needed. Per `docs/cv2_cv3_interface_spec.md` decision rule: robust
→ adopt best margin → move on. Geometry axis of the CV-2→CV-3 interface
is closed. (Domain axis was explicitly out of scope for this experiment
— see next task.)

## Completed: CV-3 Domain Validation on iToBoS (TBP) crops

**Result: BORDERLINE FAIL (78% vs 80% gate).**
`analysis/quality/cv3_domain_itobos/result.md` (2026-09-01). Script:
`scripts/validate_cv3_domain_itobos.py`. Real CV-2 B1 true-positive
detections on real iToBoS images, through the real `crop_and_normalize()`
at margin=0.25, through CV-3. No iToBoS masks exist, so this used proxy
metrics (sanity net, inconclusive on their own — no proxy showed a clean
collapse) plus a stratified n=50 manual visual audit (the actual decision
signal): 39/50 (78%) rated "reasonable," just under the pre-committed
≥80% gate.

Failure breakdown: 6/11 fails are mask **fragmentation** (scattered
disconnected blobs instead of one coherent region — the dominant mode),
3/11 near-empty/miss, 2/11 blocky border artifacts. Per the spec's
fail-branch diagnostic, checked whether this is a fixable scale/framing
issue (retry margin) vs. a genuine appearance-domain gap: box-size
analysis showed **no clean scale cutoff** (failed vs. passed crop-area
distributions heavily overlap) — this rules out "just retry with a
different margin" and points at a genuine, if partial, dermoscopic→TBP
appearance gap (texture/hair/lighting), not a geometry problem.

**Per the pre-committed spec, this does NOT auto-trigger fine-tuning.**
The fail branch explicitly stops here: fine-tuning CV-3 on TBP data would
need real TBP mask signal (none exists — no iToBoS segmentation ground
truth), and deciding to invest in collecting/generating that is its own
scoped decision with its own spec, not an automatic next step.

**Practical read:** CV-3 is usable on TBP crops as-is for continued
pipeline wiring — 78% of real CV-2 detections still get a reasonable
segmentation, and the ~22% fragmentation-dominated failure rate is now a
known, quantified limitation (same category as CV-2's known ~19%
complete-miss rate: downstream stages already have to tolerate an
imperfect upstream stage). Not a blocking defect; document and move on.

## Completed: CV-1.5 Domain Router

**Status: PASS (Stage 2). CV-1.5 exists.**
`analysis/quality/cv1_5_router/result.md` (2026-09-01). Spec:
`docs/cv1_5_router_spec.md` (staged, cheapest-first: classical heuristic
before any training, escalate only if it fails a >=90%-per-class gate).

- **Stage 1 (heuristic, `src/routing/heuristic.py`) FAILED:** 80.0%
  pre_framed / 62.0% wide_field vs. the 90% gate
  (`analysis/quality/cv1_5_router/stage1_result.md`). A single
  pigmentation-blob-salience feature conflates framing with lesion
  salience — same category of limitation as CV-2 needing a trained
  detector rather than classical CV.
- **Stage 2 (ResNet18 fine-tune, `src/routing/classifier.py`) PASSED:**
  1.000 / 1.000 on the same 150+150 held-out set (never resampled).
  Trained on RunPod GPU — **this was the project's first concrete RunPod
  trigger.** A perfect score on a proxy-labeled task (dataset identity
  standing in for verified per-image framing) was sanity-checked before
  being accepted, not assumed: manually inspected 6 held-out images
  (3/class), confirmed the classes are genuinely, obviously distinct at
  the composition level, not a labeling bug or a spurious-artifact
  shortcut. See the result doc for the full reasoning.
- Along the way, storage-constrained RunPod volume (100GB, 72% full)
  meant only iToBoS's train split was uploaded; the held-out eval's 150
  test-split images were pushed separately via the RunPod S3-compatible
  bucket rather than syncing the full 8,481-image test split (~10GB) —
  worth remembering as the pattern for any future "need a small fixed
  subset of a large remote dataset on a space-constrained pod" situation.

**Production interface:** `src/routing/classifier.py::route_image` +
`load_router_checkpoint`. Checkpoint: `checkpoints/cv1_5_router/best.pt`
(RunPod volume — not in git, matches checkpoint convention).

**Proxy-label caveat still stands** (not a blocker): evaluation measures
"PAD-UFES vs. iToBoS," not verified per-image framing. If CV-1.5
underperforms on real product images later, this gap is the first place
to look. Same category as CV-1's synthetic-only validation caveat and
CV-2's iToBoS→phone caveat.

## Completed: CV-1 → CV-4 Pipeline Assembly

**Status: assembled, measured, regression check passes.**
`analysis/product_eval/cv1_cv4_assembly/result.md` (2026-09-01).
Spec: `docs/cv1_cv4_assembly_spec.md`. Code:
`src/inference/orchestrator.py::DermaSensePipeline`. Eval:
`scripts/evaluate_pipeline_end_to_end.py`.

First time the components ran as one pipeline. `src/inference/pipeline.py`
is deliberately untouched (still the CV-4-only path, three tests depend
on its `predict(tensor)` contract); the assembly is a new class taking a
raw BGR image.

**Regression check (pre-framed, PAD-UFES 352): PASSES.** 100% per-image
agreement with CV-4 standalone on the 303 images assessed; Macro-F1
0.6351 and 31 Tier-1 errors, identical both sides when computed on the
same subset. The cv2-vs-PIL preprocessing risk did not materialize.

**Three findings pairwise validation could not have shown:**

1. **CV-1 rejects 13.6% of real clinical images** (48/352 PAD-UFES,
   plus 21.5% of wide-field iToBoS). The known "validated on synthetic
   degradation only" caveat, now measured. **Fixed the same session —
   see "CV-1 — Image Quality Gate" above; PAD-UFES now 1.42%.** Not biased toward high-risk
   (38.8% of drops vs 50.9% base rate), but it is the population the
   capture-guidance layer exists to serve.
2. **34.5% of wide-field submissions produce no assessment at all** —
   CV-1 drops 21.5%, then CV-2 finds nothing in a further 13.0%.
   Compounding attrition; neither rate alarming alone. (The 13.0% is NOT
   comparable to CV-2's documented ~19% miss: different denominator,
   measured after CV-1 already removed a fifth of the images.)
3. **Alarm fatigue on the wide-field branch** — 32.7% of assessed images
   escalate to URGENT_EVALUATION, 85% require review, driven by a 12.9%
   per-candidate high-risk rate (357 BCC in a screening population — not
   clinically plausible) amplified by most-severe aggregation over a
   mean 4.71 candidates/image. This is CV-4 running out of domain on TBP
   crops, not a reason to weaken the (correctly conservative)
   aggregation rule.

**Design decisions committed:** CV-3's mask is evidence only and never
touches CV-4's input (dependency direction — CV-4 drives risk, so a bad
mask must not be able to crop away the tissue it needed); non-assessment
is a first-class `PipelineOutcome` so "never looked" cannot read as
"looked and it's fine" (verified: the UNKNOWN count exactly equals
quality-rejected + no-candidates); multi-candidate aggregation takes the
most severe action.

**Also added:** `src/quality/capture_guidance.py` — the structured
signal layer for the suggestion engine (CV-1 quality issues first, then
CV-1.5 framing). Warns rather than blocks: a user cannot photograph a
mole on their own back, and hard-blocking wide-field would exclude the
anatomy where melanoma is most often missed in men. UI/copy decisions
remain out of scope.

**Still out of scope:** CV-5/6/7 and the convergent CV-8. The mask is
*recorded* for CV-5, not consumed by it.

## Open product question

**No user capture protocol exists yet.** Intended primary input is a
zoomed-in photo of a lesion the user is already worried about
(pre-framed); wide frames occur incidentally. This matters because the
two branches behave materially differently — see findings 2 and 3 above
— so "what we tell users to photograph" is a product decision with
direct CV consequences. Tracked here rather than left implicit.

---

## Deferred items (tracked, not forgotten)

| Item | Reason deferred | Re-entry condition |
|---|---|---|
| CV-2 tiling/SAHI refinement | **Effectively CLOSED** — gate re-scoped 2026-09-01, see `docs/cv2_status.md` | Only if wide-field becomes a primary input path AND the phone-domain gap is closed first |
| CV-2 domain validation (iToBoS→phone) | Needs real phone-image test set | **Now the operative CV-2 question** — must precede any further CV-2 refinement |
| CV-3 domain validation (TBP crops) | DONE 2026-09-01 | — (`analysis/quality/cv3_domain_itobos/result.md`) |
| CV-1 real-artifact robustness | **RESOLVED for PAD-UFES 2026-09-01** — recalibrated, rejection 13.6%→1.42% (`analysis/quality/cv1_recalibration/result.md`). iToBoS also improved (21.5%→12.4%, same recalibration) but not to near-zero — not chased further; may reflect genuine wide-field composition differences (hair, TBP-rig equipment in frame) rather than the same defect, unconfirmed | If iToBoS wide-field rejection still matters, investigate separately — was not part of criterion A's calibration set |
| CV-4 domain gap on TBP crops | **INVESTIGATED 2026-09-01** — root cause identified (crop-quality-correlated BCC/ACK confidence collapse, not general miscalibration); disclosure evidence added, no retraining/filtering (`analysis/product_eval/cv4_domain_evidence/result.md`) | If acting on the evidence is wanted (e.g. CV-6-style abstention policy), that is a separately-scoped future task |
| src/detection/ tests inside package | Inconsistency vs tests/ convention | Low priority cleanup |
| .venv torch mismatch on RunPod | System python works | Before next long pod session: pin requirements |
| `build_detector` pretrained flag no-op | Harmless for now | Before any from-scratch training attempt |
| iToBoS/UQ participant-level overlap | No exposed IDs to cross-reference | If joint evaluation ever needed |
| PAD-UFES-20 ACK lesion-level count | Not exposed in public metadata | Before any ACK-specific training claim |

---

## Key committed decisions (short form)

1. **Taxonomy:** native diagnosis classification (not 3-class Benign/Suspicious/Malignant). Risk categorization lives in CV-8, not the classifier.
2. **ISIC 2019 vs HAM10000:** use ISIC 2019 as the superset. Never add HAM10000 separately.
3. **CV-2 recall metric:** image-level (≥1 TP per lesion-containing image), not box-level. Rationale in `docs/cv2_section22_finalized.md`.
4. **CV-2 FPR metric:** per-image false-candidate burden (median/p90), not binary. Rationale ibid.
5. **CV-2 stopping rule:** max 2 further experiments (YOLO11s + optionally tiling). YOLO11s run (negative). Tiling was the last permitted attempt and is now closed unresolved-but-deprioritised (gate re-scoped 2026-09-01) — the iToBoS→phone domain gap must be answered before any further CV-2 refinement.
6. **CV-3 preprocessing:** squash resize (NOT aspect-ratio-preserving) to 512×512. The crop-normalize function MUST match this.
7. **No "loosening thresholds because experiments failed":** any metric revision must be justified by role/product reasoning independent of experiment outcomes.
8. **Leakage prevention:** patient/lesion-disjoint splits everywhere. Never split on image when lesion IDs are available.