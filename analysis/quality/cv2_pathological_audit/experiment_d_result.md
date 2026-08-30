# CV-2 Experiment D — Result: Sun-Damage Hard-Negative Oversampling

**Status:** Negative result. Intervention did not meaningfully move locked metrics.

## Hypothesis
Zero-lesion images with `sun_damage_level >= 2` (heavily freckled/sun-damaged
skin) were visually confirmed and metadata-confirmed (30% vs 5.9% rate) to
drive a disproportionate share of confident false positives. Oversampling
this subgroup 8x during training was expected to reduce zero-lesion FPR by
teaching the model to suppress freckle-driven candidates.

## Setup
- Single-variable change from B1: training list only. Architecture (YOLO11n),
  resolution (1280), epochs (50), batch (8), seed (42) all identical to B1.
- 79 hard-negative images (sun_damage_level >= 2, zero-lesion, in train split)
  duplicated 8x in train_oversampled_sundamage.txt (7331 lines vs 6778).
- Effective per-epoch hard-negative share: 8.6%.

## Result (val split, conf=0.25)
| Metric            | B1 (1280) | D (oversampled) | Change   | Target |
|-------------------|-----------|-----------------|----------|--------|
| Lesion recall     | 0.5436    | 0.5448          | +0.0012  | >=0.95 |
| Zero-lesion FPR   | 0.2063    | 0.1948          | -0.0115  | <=0.05 |
| 10+ lesion recall | 0.4766    | 0.4830          | +0.0064  | >=0.90 |

## Conclusion
All three changes are within run/seed noise. FPR improved by ~1.2 points, far
short of the gap to target (~20% vs <=5%). Targeted hard-negative oversampling
of a ~79-image subgroup does not meaningfully address a problem where FPR is
~4x above target and recall ~40 points below it.

## Decision
Do NOT tune this lever further (larger oversample factor, wider filter). The
consistent signal across the CV-2 arc — no usable confidence threshold exists
(sweep), resolution increase (B0->B1) had no effect on locked metrics, and now
targeted oversampling (D) had no effect — points to a fundamental capability/
scale gap, not something reachable by data/threshold/sampling adjustments.

Next lever (different in kind, not a tweak of D): larger backbone (YOLO11s/m)
to test the under-capacity hypothesis, and/or tiled inference (SAHI-style) to
attack the extreme small-object scale (median lesion normalized area 0.00097).
