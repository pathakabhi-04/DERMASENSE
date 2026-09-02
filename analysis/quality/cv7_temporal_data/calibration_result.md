# CV-7 Ruler Calibration — Result

**Status:** Implemented and gated. Confident-calibration coverage
measured honestly at 4.0% on a random sample — low, but the module is
designed so that number is safe to ship: it fails loudly rather than
guessing. Decision (user, 2026-09-02): ship as v1, revisit detection
sensitivity later if size measurement proves valuable enough to invest
further.

## What was built

`src/temporal/calibration.py::calibrate(image_bgr) -> RulerCalibration`.
Detects the ruler's horizontal tick marks via a probabilistic Hough
transform (chosen after two other approaches — darkness+shape blob
heuristics, then the same with HSV saturation filtering — failed to
generalize across cameras; Hough directly targets "short straight
segment at a known angle," which is what a tick actually is). Computes
px/mm from the median tick-to-tick spacing, gated on three checks:
enough ticks found, spacing consistent (not just numerous), and the
result falling inside a corroborated plausibility band. Any gate
failure returns `confident=False, px_per_mm=None` — never a guessed
number.

## The 1mm/tick assumption, and why it's trusted

Not stated in the dataset's own documentation (checked: `FurtherInformation.txt`
in the archive only points back to the eSpace listing, which doesn't
cover it; the paper's own methods text was inaccessible — both PMC and
ResearchGate blocked automated fetches). Corroborated instead by two
independent sources landing on the same number:

- Canfield's own product page for the **VEOS SLR** (one of the two
  cameras actually used in this study, confirmed via the paper's
  abstract) states a **270 pixels/mm** contact-plate scale.
- This module's own direct pixel measurement, done before writing any
  detection code, found ticks spaced ~266px apart on a *Canon*-camera
  image (a different camera from the VEOS) — within 1.5% of Canfield's
  spec on unrelated equipment.

`EXPECTED_PX_PER_MM_RANGE = (220, 320)` encodes this as a falsifiable
plausibility check, not a hardcoded value ever assumed without
measuring per image.

## Coverage measurement (200-image random sample, seed 42)

| | |
|---|---|
| Confident calibrations | **8 / 200 (4.0%)** |
| px/mm when confident | mean 264.8, median 264.2, range [262.2, 269.0] |

**The confident results are tight and consistent** — a 6.8px spread
across 8 independent images, landing almost exactly on the
manufacturer-corroborated ~265px/mm. This is the reassuring half of the
finding: when detection succeeds, it is not landing on the right answer
by chance.

**The gap is detection sensitivity, not correctness.** Failure
breakdown:

| Reason | Count |
|---|---|
| Fewer than 2 tick candidates detected at all | 125 (62.5%) |
| Spacing found but too irregular | 48 (24.0%) |
| Only 1 plausible tick-to-tick gap | 8 (4.0%) |
| Only 0 plausible gaps among candidates | 7 (3.5%) |
| Only 2 plausible gaps (need ≥3) | 4 (2.0%) |

Nearly two-thirds of failures are "the ruler wasn't found at all" —
consistent with what a small manual dev-set check suggested: hair
density, lesion-boundary curves crossing the ruler region, and
non-fixed ruler framing (these are handheld shots, not a rigid jig —
tested and rejected the hypothesis that the ruler sits at a fixed pixel
position across images) all reduce how often the tick pattern is
cleanly legible to a generic line detector.

## What was tried and rejected before Hough line detection

In order, each evaluated against real images before moving on (not
guessed at):

1. **Darkness threshold + connected-component shape filter** (wide,
   short blobs). Worked on one hand-picked image, but picked up the
   "mm" text glyph's strokes as false ticks, and failed on images with
   fainter tick contrast.
2. **Adding an HSV saturation constraint** to reject brownish hair
   (vs. near-black ticks). Rejected: measured real tick pixels had
   saturation up to ~130 in places (JPEG chroma noise on the etched
   mark), above the threshold that would have been needed to reject
   hair — the constraint eliminated genuine ticks along with hair.
3. **Narrowing the x-search-region**, on the hypothesis that a
   camera-mounted ruler holds a fixed frame position. Tested directly:
   made some images better and others worse, disproving the hypothesis
   — these are handheld shots with real framing variance.

## Decision

Ship the confidence-gated detector as-is. A candidate whose calibration
fails gets `NO_PRIOR_DATA` for the size dimension specifically — color
and border deltas (which don't need the ruler) are unaffected. Coverage
is explicit and measured (this document), not hidden inside a silently
degraded average. Improving detection sensitivity (e.g. template
matching against the ruler's actual visual pattern, which may be far
more consistent than its frame position) is a separately-scoped
follow-up, not attempted now — per the anti-rabbit-hole boundary in
`docs/cv7_temporal_technical_spec.md`.
