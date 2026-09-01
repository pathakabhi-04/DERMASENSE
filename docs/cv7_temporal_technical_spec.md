# CV-7 Temporal — Technical Spec

**Status:** Committed before implementation, per the same discipline as
every prior spec. Written against the staged 30-participant sample
(`analysis/quality/cv7_temporal_data/result.md`) after actually
inspecting the images and metadata — not assumed.

## What was inspected before writing this

- **Images**: 6000×4000 JPEG, genuine dermoscopic close-ups (through a
  dermatoscope) — matching CV-3's actual training domain (ISIC 2018
  Task 1), unlike CV-3's measured struggles on iToBoS wide-field crops.
- **A physical mm ruler is etched directly into the frame** (visible,
  consistent bottom-left position in every sample checked). This
  enables real-world-unit size measurement via classical image
  processing — no training needed for the size dimension at all.
- **Multiple cameras were used**: Canon EOS Rebel T6i (82%), Veos SLR
  (17%), plus minor others (`UNKNOWN`, `#N/A`, `VISIOMED 16`, <1%
  combined). This rules out a single global mm-per-pixel constant —
  the ruler must be read **per image**.
- **Real clinical ground truth exists**: a `Diagnosis` field per
  lesion (benign, melanoma, basal cell carcinoma, squamous cell
  carcinoma, actinic keratosis, nevus, seborrheic keratosis, etc.),
  96.5% benign dataset-wide (34,668/35,909) — consistent with a
  surveillance cohort where most monitored lesions stay benign.
- **Diagnosis is a per-lesion outcome label, not a per-visit-in-time
  assessment**: confirmed 0/9,382 lesions have a diagnosis that
  differs across their own visits. This means CV-7's temporal-change
  measurement and the diagnosis label are independent signals —
  exactly what's needed to test, with real ground truth, whether
  measured change correlates with malignant outcomes.
- **Malignant-outcome coverage is thin in the staged sample**: our
  30-participant sample has only 4 malignant-outcome lesions (3 BCC, 1
  SCC) out of 643 lesions. Dataset-wide there are 147 malignant-outcome
  lesions, 99 with ≥2 visits (usable for change measurement), across 57
  participants — see "Storage and the malignant-enrichment pull" below.

## Design: Stage 1 is classical and deterministic, not learned

Mirroring CV-1.5's cheapest-first structure (a classical heuristic
before any training) and CV-1's recalibration discipline (calibrate
thresholds once against real data, document the calibration): given the
ruler and CV-3's already-matching training domain, a fully deterministic
pipeline is possible for v1 — **no new training required at all**.

1. **Per-image ruler calibration.** Detect the mm tick marks in the
   known corner region (classical CV — line/tick detection), compute a
   pixel-to-mm scale factor for that specific image. Never assume a
   fixed scale, per the camera-diversity finding above.
2. **Lesion segmentation.** Reuse CV-3 directly
   (`src/segmentation/model.py::build_model()` +
   `checkpoints/cv3_512/best.pt`) — no new segmentation model. This is
   real dermoscopic imagery, CV-3's actual training domain, so this is
   expected to generalize far better here than it did on iToBoS TBP
   crops (measured 78% reasonable there; no reason to expect worse
   here, though this gets verified, not assumed — see Evaluation).
3. **Per-visit measurement**, from the segmented mask + that image's
   calibration: diameter/area in mm (real units, not pixels), mean
   color in Lab space (perceptually uniform, unlike RGB), contour
   irregularity (compactness — `perimeter² / (4π·area)`, a standard
   border-irregularity measure).
4. **Pairwise delta computation** between same-lesion visit pairs:
   magnitude + direction per feature (size/border/color), using the
   earlier visit as the reference for direction.
5. **Verdict assignment** (`STABLE | GROWING | SHRINKING |
   CHANGED_COLOR | NO_PRIOR_DATA`, matching the locked contract in
   `docs/cv7_temporal_rag_integration_spec.md`) via thresholds
   calibrated once against the measured deltas in the staged sample —
   not arbitrary, not iteratively tuned after the fact.

**Explicitly deferred to a Stage 2 (only if Stage 1 proves
insufficient):** a learned model (e.g. a Siamese/paired-embedding
network) trained end-to-end on image pairs. Not attempted
preemptively — same escalation discipline as CV-1.5's Stage 1→Stage 2
gate. If Stage 1's classical measurements can't discriminate malignant
from benign change patterns at all, that failure mode — not a vague
sense that "a bigger model would help" — is what would justify Stage 2.

## Evaluation plan (two separate questions, two separate data needs)

**1. Pipeline correctness** — does the classical measurement pipeline
work at all? Answerable now, on the existing 30-participant/643-lesion
sample: run the pipeline, visually audit a bounded sample of computed
deltas against the actual image pairs (same visual-audit discipline as
every other component this session — CV-1.5 Stage 2, CV-4 domain
evidence, CV-5's Grad-CAM). Needs no malignant examples; it's testing
arithmetic and segmentation quality, not clinical validity.

**2. Clinical signal validity** — do GROWING/CHANGED_COLOR verdicts
occur more often, or more strongly, on malignant-outcome lesions than
benign ones? This is the actual question CV-8 needs answered, and it is
falsifiable against real ground truth (the `Diagnosis` field). It
needs the malignant-enrichment pull below — 4 malignant lesions is not
enough to test anything.

**No pass/fail gate is pre-committed here for question 2** — the
malignant-enriched data isn't staged yet (see below), so committing a
numeric threshold before knowing the achievable sample size would be
premature. That gate gets set once that data is in hand.

## Storage and the malignant-enrichment pull

Direct answer to "is the current subset enough": **enough to build and
validate the pipeline's correctness; not enough to validate whether it
means anything clinically.**

Rather than a full volume resize, the efficient fix is a **second,
targeted, small pull**: the 57 participants (dataset-wide) who carry at
least one longitudinal (≥2-visit) malignant-outcome lesion —
**11.74GB, 6,365 images, 99 malignant lesions with visit pairs**.
Combined with the existing sample (~16-17GB total, accounting for
participant overlap), this still fits within current free space on both
local disk (~18GB free) and the RunPod volume (~25-30GB free) — **no
resize needed for this**, because the actual bottleneck was never raw
image volume, it was malignant-example scarcity, and that's fixable by
targeting instead of scaling.

A full resize (~70-80GB more, ~$5-6/month) stays deferred until there's
a concrete reason to need it — e.g., Stage 2 (a learned model) turning
out to be necessary, which would want the full 331-participant set the
way CV-1.5's Stage 2 needed the full PAD-UFES/iToBoS train splits.

## Anti-rabbit-hole boundary

One classical pipeline, calibrated once against the staged sample,
evaluated once against the malignant-enrichment pull once it's staged.
Stage 2 is not attempted preemptively — only if Stage 1's own
evaluation shows a genuine, specific insufficiency, not a general sense
that a learned model "would probably do better."

## Files (implementation, next step after this spec)

- `src/temporal/calibration.py` — ruler detection, per-image
  pixel-to-mm scale factor
- `src/temporal/measurement.py` — per-visit size/color/border
  measurement from a CV-3 mask + calibration
- `src/temporal/delta.py` — pairwise delta computation + verdict
  assignment against calibrated thresholds
- `scripts/calibrate_cv7_thresholds.py` — one-time threshold
  calibration against the staged sample, following the same pattern as
  `scripts/calibrate_cv1_resolution.py` and
  `scripts/calibrate_cv6_temperature.py`
- `tests/test_temporal.py` — unit tests on synthetic masks/rulers (no
  data needed) + an integration test against the staged sample
