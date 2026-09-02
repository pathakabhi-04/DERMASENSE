# CV-7 Measurement — Result

**Status:** Implemented. Built on a verified foundation: CV-3
produces usable masks on real UQ Longitudinal dermoscopic images at a
5.0% degenerate-empty rate (below), far better than its ~22%
fragmentation rate on iToBoS TBP crops — confirming the technical
spec's expectation that this domain matches CV-3's actual ISIC
training distribution.

## Foundational check: does CV-3 segment this domain well?

Before writing any measurement logic, ran CV-3
(`checkpoints/cv3_512/best.pt`) directly on real UQ Longitudinal
images (no CV-2 crop step — these are already lesion-centric
close-ups) and measured mask degeneracy the same way
`mask_evidence()` does elsewhere in the codebase.

**4-image visual check** (dev sample spanning both cameras): 3/4 tight,
accurate segmentations — including one image where CV-3 correctly
found two separate lesions in-frame. 1/4 a complete miss on a faint,
low-contrast lesion (`HighRisk99_visit1`).

**100-image random sample (seed=7, independent from calibration's
seed=42):**

| | |
|---|---|
| Degenerate empty (fg=0) | 5 / 100 (5.0%) |
| Degenerate full (fg>0.95) | 0 / 100 (0%) |
| fg_frac | mean 0.0601, median 0.0384, p10 0.0084, p90 0.1098 |

Conclusion: CV-3 is trustworthy enough on this domain to build
measurement logic directly on top of its mask, with a simple gate for
the empty-mask minority (`valid=False`) — no detection/cropping step
needed first, unlike the extensive struggle required for ruler
detection.

## What was built

`src/temporal/measurement.py::measure_lesion(image_bgr, mask,
calibration) -> LesionMeasurement`. Two independent gates, mirroring
`calibration.py`'s fail-loud design:

- **`valid`**: was a lesion found at all (mask non-empty after
  largest-connected-component isolation)? If not, nothing is
  measurable — no size, color, or border fields.
- **`diameter_mm` / `area_mm2` being `None`** (even when `valid`): was
  ruler calibration confident for this image? `area_fraction`
  (pixel-space) and `compactness` (border irregularity) are
  scale-invariant by construction and always available when `valid` —
  only real-unit size needs the ruler.

## Design decisions made explicitly

- **Multi-blob masks → largest connected component only.** The
  4-image check found a real case (`HighRisk78`) where CV-3 correctly
  segmented two separate lesions in one frame. Since the dataset's
  filename convention is one `Lesion{N}` per image, a second blob is
  treated as an incidental nearby freckle/mole, not the lesion of
  interest — only the largest component is measured.
- **Diameter = minimum-enclosing-circle diameter**, not an
  area-equivalent diameter (`2*sqrt(area/pi)`). Chosen because
  clinical size tracking (the ABCDE "D") means the lesion's greatest
  diameter, which a minimum-enclosing-circle approximates efficiently
  without full rotating-calipers max-pairwise-distance computation.
- **Border compactness = `perimeter² / (4π·area)`**, the same standard
  formula named in the technical spec — 1.0 for a perfect circle,
  higher for irregular borders. Unit test confirms it lands in
  [1.0, 1.3) on a filled circle (pixelated-contour perimeter
  overestimates a discretized circle's true perimeter, so exact 1.0
  isn't expected even for genuinely round input).
- **Color = mean Lab**, not mean RGB, per the technical spec (Lab is
  perceptually uniform, so a fixed distance in Lab space corresponds
  to a roughly fixed perceived color difference, unlike RGB).
- **Mask resized to the original image's resolution** (not measured at
  CV-3's 512×512 output resolution) via nearest-neighbor
  interpolation, so pixel-space measurements share the same coordinate
  system as `calibrate()`'s px/mm (measured directly on the original
  image).

## Testing

`tests/test_measurement.py` — 6 synthetic unit tests (known-geometry
circle mask: area/compactness/color math, calibrated vs.
non-confident real-unit gating, empty-mask handling, multi-blob
component isolation, mask-resize correctness) + 1 integration test
running the real CV-3 checkpoint on a known real image from the
staged sample (regression-pinned, skipped if the checkpoint or source
zip is unavailable). All 7 pass.
