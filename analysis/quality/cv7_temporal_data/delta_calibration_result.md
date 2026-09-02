# CV-7 Delta Thresholds — Calibration Result

**Status:** Border and color thresholds calibrated against real measured
deltas. Size (growth) threshold left explicitly provisional — the data
needed to calibrate it does not exist in usable quantity yet, and that
scarcity is itself an expected, already-documented consequence of
calibration's 4.0% coverage, not a new problem.

## Method

`scripts/calibrate_cv7_thresholds.py`. Bounded, seeded (seed=11) sample
of **300 visit pairs** — one pair (first two visits) per staged lesion,
drawn from the 84 currently-staged participants (30-participant sample +
54-participant malignant enrichment). For each pair: ran calibration +
CV-3 segmentation + `measure_lesion` on both images, then
`compute_delta`. This does not touch the full 331-participant dataset —
same bounded-sample discipline as every other CV-7 measurement so far.

## Results

| | |
|---|---|
| Pairs sampled | 300 |
| Both masks valid (border/color computable) | 280 (93.3%) |
| Both visits confidently calibrated (size computable) | **1 (0.3%)** |
| Malignant-outcome lesions in sample | 19 |

**abs(border_delta) percentiles** (n=280): p50=0.15, p75=0.86, p90=3.16,
p95=4.86

**color_delta (CIE76 Lab distance) percentiles** (n=280): p50=8.88,
p75=16.87, p90=23.98, p95=31.63

**abs(size_pct_change)**: n=1, value=6.13% — a single data point, not a
distribution.

**Malignant vs. benign, descriptive only (no gate committed — see
"Why no size threshold" below and the technical spec's own statement
that no pass/fail gate is pre-committed for the clinical-validity
question):**

| | malignant (n=19) | benign (n≈261) |
|---|---|---|
| mean abs(border_delta) | 3.08 | 0.89 |
| mean color_delta | 13.69 | 12.16 |

Malignant-outcome lesions show a notably larger mean border-shape delta
in this sample. This is suggestive, not confirmatory — n=19, a single
consecutive-visit pair each, no correction for multiple comparisons —
and is recorded as an observation for whoever builds CV-8's evidence
weighting, not as a validated finding.

## Thresholds set

- **`COMPACTNESS_DELTA_THRESHOLD = 3.0`** (≈p90 of real measured
  border-shape change). Used only as a magnitude normalizer — per the
  locked JSON contract (`docs/cv7_temporal_rag_integration_spec.md`),
  border has no headline verdict of its own (no `CHANGED_BORDER`
  state); it is always exposed as `per_feature_deltas.border` evidence.
- **`COLOR_DELTA_E_THRESHOLD = 24.0`** (≈p90 of real measured color
  change). Set deliberately high, not from a color-science convention:
  the *median* observed color delta (8.88) already exceeds what a
  clinically-naive threshold like "8" would have used — most
  visit-to-visit Lab distance in this dataset is very likely lighting/
  camera/white-balance variation across different photo sessions, not
  biological pigment change (multiple cameras were used per the
  technical spec, and visits are handheld shots taken on different
  days). A low threshold here would flag `CHANGED_COLOR` on capture-
  condition noise, not on real change. p90 was chosen over p95 as a
  middle ground given no independent way (yet) to separate lighting
  noise from real change in this dataset. **This is a documented
  limitation, not a solved problem** — see "Known limitation" below.

## Why no size (growth) threshold was set

`GROWTH_PCT_THRESHOLD` and `GROWTH_ABS_MM_FLOOR` remain the placeholder
values from initial implementation (20.0%, 0.5mm) — explicitly flagged
in `src/temporal/delta.py` as **not calibrated from data**, unlike
border and color. The reason is structural, not an oversight: a
size-change verdict needs BOTH visits to have a confident ruler
calibration, and calibration's own measured single-image confident rate
is 4.0%. Two independent images each at ~4% gives a compounding
expected rate of ~0.16%, consistent with what was actually observed:
**1 double-confident pair out of 300 (0.3%)**.

Chasing a properly-calibratable sample size (even ~30 double-confident
pairs, for a very rough percentile estimate, would need roughly
300/0.003 ≈ 10,000 pairs processed) is not attempted now — it would mean
running CV-3 + calibration across most of the entire 8,751-image staged
corpus for a feature that stays structurally rare regardless of sample
size, since the bottleneck is ruler-detection sensitivity, not sample
size. This is the same anti-rabbit-hole boundary already applied to
ruler-detection sensitivity itself
(`analysis/quality/cv7_temporal_data/calibration_result.md`): improving
it is a separately-scoped follow-up, not attempted here.

## Known limitation

Color-delta comparison across visits is confounded by capture
conditions (camera model, lighting, white balance) that this module
cannot currently separate from real biological color change. The
p90-based threshold is a conservative mitigation (favoring missed real
changes over false alarms on lighting noise), not a fix. A future
improvement (not attempted now) could normalize color against a
reference patch in-frame (e.g. the ruler itself, or a fixed-reflectance
region) if one exists consistently across images — untested, flagged
for later.

## Decision

Ship `src/temporal/delta.py` with border/color thresholds set from this
calibration and the size threshold left explicitly provisional and
documented as such (never silently presented as equally well-founded).
A `GROWING`/`SHRINKING` verdict will be rare in practice — bounded by
calibration's own 4% ceiling on each side — which is consistent with
`NO_PRIOR_DATA`-for-size being the expected, safe, common outcome for
the size dimension specifically, exactly as already decided for
calibration.py itself.
