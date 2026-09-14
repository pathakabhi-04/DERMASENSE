# Abstention: a real signal, below its bar — and it belongs at capture time

**Date:** 2026-09-15
**Script:** `scripts/evaluate_abstention.py` (method and decision rule
pre-committed in its docstring)
**Data:** the 676 assessed external images, of which **37 are dangerous
misses** — melanomas the pipeline assessed and classified NEV/SEK, so
sent to `MONITOR` instead of a clinician.

## The question

Not "can we classify better" — four investigations closed that. Instead:
**can the system tell when it is about to be wrong, and say so?** A
dangerous miss converted into "I cannot assess this" stops being false
reassurance and becomes an honest non-answer, which the pipeline already
treats as a first-class outcome.

## Model confidence does not know when it is wrong

On those same melanomas:

| signal | routed (median) | missed (median) |
|---|---:|---:|
| `calibrated_confidence` | 0.7031 | **0.7028** |
| `confidence` | 0.7902 | 0.7729 |

No separation at all. That is worth stating plainly: the calibration work
produces a number that is useful for *ranking* but carries no information
about *this particular prediction being wrong*. Any abstention policy
keyed on the classifier's own confidence would do nothing.

So the test used only signals computed **before** classification — CV-1
quality and CV-3 mask evidence.

## Result: one real signal, and it does not clear the bar

Thresholds were chosen on half the data and reported on the untouched
half. Then, because the two halves disagreed sharply (33% vs 53%), the
whole procedure was repeated across **40 random splits**:

| signal | misses caught (median) | abstention | lift vs null | beats null |
|---|---:|---:|---:|---:|
| `mask_area_fraction` | **36.8%** | 22.8% | **1.62×** | **98%** of splits |
| `crop_contrast` | 21.1% | 22.8% | 0.92× | 42% |
| `quality_score` | 13.2% | 20.3% | 0.65× | 10% |
| `crop_blur` | 0.0% | 2.7% | — | 35% |

**Verdict: does not clear the pre-committed bar.** The rule required
≥50% of dangerous misses caught at ≤25% abstention. `mask_area_fraction`
hit 52.6% on the single pre-registered split — and that was **split
luck**: across 40 splits its median is 36.8% and its 90th percentile is
53.2%. Reporting the one passing split would be precisely the
result-selection this project has spent four experiments avoiding.

Two things are nonetheless true and worth keeping:

- **`mask_area_fraction` is a genuine signal**, not noise: 1.62× lift,
  beating random abstention in 98% of splits. It is real and it is
  insufficient.
- **`crop_contrast` is not a signal**, despite having the largest median
  gap in the exploratory look that motivated this whole check (0.94
  routed vs 0.60 missed). Under a held-out threshold it delivers 0.92×
  — *worse than random*. A difference in medians is not a usable
  decision rule, and that gap is exactly why the split existed.

## What the surviving signal actually means

The misses concentrate on images where CV-3 finds a **small lesion
relative to the frame** — under-zoomed, badly-framed photographs. That is
not primarily an abstention finding. It is a **capture** finding:

> The pipeline's dangerous failures are disproportionately on photos the
> user could have taken better, and the pipeline can tell at capture
> time, before any classification happens.

CV-1 already computes capture suggestions and the product already has a
"move closer / the lesion is cut off" surface. Routing this signal there
— *prevent* the bad photo — is strictly better than abstaining after the
fact, because it gives the user something to do about it.

As a post-hoc abstention gate it is a poor trade: ~37% of dangerous
misses caught, at 23% of all images abstained including ~25% of
correctly-handled benign lesions. As a capture prompt it costs the user
one retake and has no false-abstention cost at all.

## Decision

- **No abstention gate is added** to CV-8. The rule did not clear its
  pre-committed bar, and the honest reading of 40 splits says the single
  passing split was luck.
- **The finding is handed to CV-1 capture assistance instead**, where it
  is actionable and cheap. That is part of the narrow product, which is
  what is shipping.
- **Confidence-keyed abstention is closed**: the classifier's confidence
  carries no information about its own errors here.

## Caveats

- 37 dangerous misses total, 19 per held-out half. Every figure carries a
  wide interval; the 40-split median is more trustworthy than any single
  split, which is the point of having run them.
- Single-signal thresholds only. A multivariate rule fitted on 37
  positives would overfit, and refusing to fit one was pre-committed.
- Measured on three clinical-photo sources, not on phone photos from the
  deployment population — the same standing caveat as everything else in
  this series.
