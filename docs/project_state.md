# DermaSense — Project State Document

**Last updated:** end of CV-2 baseline freeze session (Aug 2026).
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
**Status: COMPLETE (baseline)**
- Quality assessment and guidance implemented in `src/quality/`.
- Robustness validation done (`analysis/quality/cv1_robustness/`).
- Known limitation: validated against synthetic degradation; not yet
  tested against real PAD-UFES-20 artifacts (marker ink, hair, glare).
  Flagged as a future validation task, not blocking.

### CV-1.5 — Domain Router
**Status: DESIGNED, NOT IMPLEMENTED**
- Decision: routes on framing (pre-localized vs wide-field), not modality.
- PAD-UFES-20 is pre-framed (visually confirmed), so it takes the
  pre-framed path directly to CV-3.
- Implementation needed before end-to-end pipeline exists.

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

**Deferred refinement:** tiling/SAHI (one experiment permitted).
Re-entry gate: run ONLY IF end-to-end evaluation (through CV-8) shows
CV-2 miss rate is the dominant pipeline failure. See `docs/cv2_status.md`.

**Constraint on downstream development:** do NOT develop CV-3 on CV-2's
real detection crops. CV-3 develops on ISIC ground-truth crops. The
CV-2→CV-3 interface is validated separately (see current task below).

### CV-3 — Lesion Segmentation
**Status: BASELINE COMPLETE**
- U-Net from scratch, `src/segmentation/model.py` (`build_model()`).
- Trained on ISIC 2018 Task 1 (512×512, binary masks, BCE+Dice loss).
- **Baseline: Dice 0.8640, IoU 0.7851** (frozen test set).
- Checkpoint: `checkpoints/cv3_512/best.pt` (locally and on volume).
- Input: 512×512, BGR→RGB, /255, CHW. Squash resize (NO aspect ratio
  preservation). Loads via:
  `build_model()` + `torch.load(...)["model_state_dict"]`.
- Known domain limitation: trained on dermoscopic (ISIC). TBP/smartphone
  domain gap not yet evaluated.

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
- Checkpoints: `checkpoints/isic2019_resnet50_weighted_best.pt` (ISIC
  baseline), `checkpoints/pad_ufes_c1_partial_finetune_best.pt` (transfer).

### CV-5 — Explainability
**Status: NOT STARTED**
- `src/explainability/__init__.py` exists (empty).

### CV-6 — Uncertainty
**Status: NOT STARTED**
- `src/uncertainty/__init__.py` exists (empty).

### CV-7 — Temporal ("What Changed?")
**Status: NOT STARTED**
- `src/temporal/__init__.py` exists (empty).
- Dataset: UQ Longitudinal (35,909 dermoscopic images, 7,038 lesions,
  340 participants, 2–7 time points). On network volume.

### CV-8 — Risk Engine / Severity
**Status: PARTIALLY IMPLEMENTED**
- `src/risk/action_mapping.py` and `src/risk/safety_gate.py` exist.
- `src/inference/pipeline.py` wires CV-4 → risk engine (Phase 4 safety
  gate). This is the CV-4-only pipeline, not the full CV-1→CV-8 pipeline.
- The full convergent CV-8 (CV-5 + CV-6 + CV-7 all feeding it) is not
  yet implemented — it's the eventual target.

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

## Current task (immediate next)

Not yet chosen. Candidates per the deferred-items table and CV component
status: CV-1.5 domain router (designed, not implemented — blocks having
an actual end-to-end pipeline), or continuing CV-4/CV-8 wiring now that
CV-2→CV-3 interface (geometry: robust; domain: measured, usable-with-caveat)
is closed out. Decide next based on product priority, not on chasing the
CV-3 TBP fragmentation finding further (per the anti-rabbit-hole
discipline — see `[[feedback_bounded_experiments]]` memory / the
SCC/BCC precedent).

---

## Deferred items (tracked, not forgotten)

| Item | Reason deferred | Re-entry condition |
|---|---|---|
| CV-2 tiling/SAHI refinement | 0.81 floor acceptable for MVP | End-to-end eval shows CV-2 miss rate is dominant failure |
| CV-2 domain validation (iToBoS→phone) | Needs real phone-image test set | After CV-1.5 domain router exists |
| CV-3 domain validation (TBP crops) | No iToBoS masks for Dice | After geometry validation passes; scoped separately |
| CV-1 real-artifact robustness | Validated on synthetic only | Before any clinical deployment claim |
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
5. **CV-2 stopping rule:** max 2 further experiments (YOLO11s + optionally tiling). YOLO11s already run (negative). Tiling is the last permitted attempt.
6. **CV-3 preprocessing:** squash resize (NOT aspect-ratio-preserving) to 512×512. The crop-normalize function MUST match this.
7. **No "loosening thresholds because experiments failed":** any metric revision must be justified by role/product reasoning independent of experiment outcomes.
8. **Leakage prevention:** patient/lesion-disjoint splits everywhere. Never split on image when lesion IDs are available.