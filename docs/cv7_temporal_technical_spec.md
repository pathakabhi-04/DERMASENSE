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

## Update (2026-09-02): calibration implemented, coverage measured

`src/temporal/calibration.py` is built and gated
(`analysis/quality/cv7_temporal_data/calibration_result.md`). Real,
measured confident-calibration rate on a 200-image random sample:
**4.0%** — low, but the confident results are tight and consistent
(262.2–269.0 px/mm, corroborated by an independent manufacturer spec),
confirming the gap is detection *sensitivity* (mostly "ruler not found
at all," from hair/lesion-curve occlusion and non-fixed handheld
framing), not a wrong assumption or a miscalibrated one. Decision: ship
as designed — a failed calibration returns `NO_PRIOR_DATA` for the size
dimension only (color/border deltas don't need the ruler), never a
guessed value. Improving detection sensitivity (e.g. template matching
against the ruler's visual pattern) is a separately-scoped follow-up,
not attempted now, per this spec's own anti-rabbit-hole boundary.

## Update (2026-09-02): measurement implemented, CV-3 domain fit verified

`src/temporal/measurement.py` is built
(`analysis/quality/cv7_temporal_data/measurement_result.md`). Before
writing it, verified CV-3 actually segments this domain well — a
100-image random sample (seed=7) measured **5.0% degenerate-empty
masks, 0% degenerate-full**, far better than the ~22% fragmentation
rate measured on iToBoS TBP crops, confirming this domain matches
CV-3's training distribution as expected. `measure_lesion()` mirrors
calibration's fail-loud design with two independent gates: `valid`
(was any lesion mask found) and calibration-confidence (whether
`diameter_mm`/`area_mm2` can be real-unit; `area_fraction` and
`compactness` are scale-invariant and always available when valid). A
multi-blob mask (a real case found during validation) is resolved by
keeping only the largest connected component, since the dataset names
one lesion per image.

## Update (2026-09-02): delta/verdict implemented, thresholds calibrated

`src/temporal/delta.py` and `scripts/calibrate_cv7_thresholds.py` are
built. Full result:
`analysis/quality/cv7_temporal_data/delta_calibration_result.md`. A
300-pair bounded sample (seed=11) of the staged data measured real
deltas: border and color thresholds were set from this (p90 of 280
pairs with both masks valid: border ≈3.16, color ≈23.98 CIE76). The
**size (growth) threshold could not be calibrated the same way** — only
1/300 pairs (0.3%) had confident ruler calibration on BOTH visits, the
direct compounding consequence of calibration's own 4.0% single-image
rate (0.04² ≈ 0.16%, consistent with what was observed). Rather than
chase enough double-confident data to calibrate it properly (would mean
processing most of the 8,751-image staged corpus for a feature that
stays structurally rare regardless), `GROWTH_PCT_THRESHOLD` ships as an
explicitly-flagged provisional placeholder, not silently presented as
equally well-founded as the other two. A secondary finding, descriptive
only (n=19 malignant lesions, not a validated result): malignant-outcome
lesions showed notably larger mean border-shape delta (3.08) than
benign (0.89) in this sample.

Also discovered and documented as a known limitation: color-delta
comparison is confounded by capture conditions (camera/lighting/
white-balance differ across visits, taken on different days) — the
median observed color delta (8.88) already exceeded what a naive
threshold would have used, so the calibrated threshold is set high
(conservative toward missed real change over false alarms on lighting
noise), not treated as a solved problem.

## Files

- `src/temporal/calibration.py` — DONE. Ruler detection, per-image
  pixel-to-mm scale factor, confidence-gated (see above).
- `src/temporal/measurement.py` — DONE. Per-visit size/color/border
  measurement from a CV-3 mask + calibration (see update above).
- `src/temporal/delta.py` — DONE. Pairwise delta computation + verdict
  assignment (see update above).
- `scripts/calibrate_cv7_thresholds.py` — DONE. One-time threshold
  calibration against the staged sample, following the same pattern as
  `scripts/calibrate_cv1_resolution.py` and
  `scripts/calibrate_cv6_temperature.py`.
- `tests/test_temporal.py`, `tests/test_measurement.py`,
  `tests/test_delta.py` — unit tests on synthetic data (no data
  needed) + integration tests against the staged sample.
