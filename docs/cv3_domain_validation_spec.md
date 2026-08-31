# CV-3 Domain Validation on iToBoS (TBP) Crops — Spec

**Status:** Committed before running (pre-registered criteria), per the
same discipline as CV-2 Experiments D/E and the CV-2→CV-3 geometry
interface validation.

## Prerequisite (satisfied)

`analysis/quality/cv2_cv3_interface/validation_run.txt` — geometry
interface validation — passed:

- Harness self-check (margin=1.0/offset=0.0): Dice 0.8631, recovers
  baseline (0.8640). Harness trusted.
- Realistic crop (margin=0.25/offset=0.1): Dice 0.8742 ≥ 0.75 → **ROBUST**.
- Worst cell in the entire 15-cell grid (margin=0.0/offset=0.2): Dice
  0.8160, still ≥ 0.75. No collapse anywhere in the swept range.
- `src/inference/crop_normalize.py` already defaults `margin=0.25`,
  which the grid confirms is also the best- (or near-best-) performing
  value (0.8776 at offset=0.0, highest mean_dice in the full grid). No
  code change needed — the shipped default is empirically validated,
  not just assumed.

Per `docs/cv2_cv3_interface_spec.md`: "Robust → adopt the crop margin
that maximizes Dice as the pipeline default. Move on." Done — the
geometry axis of the CV-2→CV-3 interface is closed.

This satisfies the re-entry condition for the deferred item **"CV-3
domain validation (TBP crops)"** in `docs/project_state.md`
("After geometry validation passes; scoped separately").

## What this validates

CV-3 was trained and Dice-measured exclusively on ISIC 2018
(dermoscopic: cropped, contact-scope, uniform lighting). In production
it receives crops from CV-2, which was trained on iToBoS (wide-field
TBP: phone/tripod-distance, variable lighting, hair, scale). The
geometry experiment above isolated crop *shape*; it deliberately held
domain constant (ISIC only). This experiment asks the domain question:
does CV-3 produce coherent lesion masks on real iToBoS crops, or does
the dermoscopic→TBP domain gap break segmentation quality?

## Why this can't be a Dice experiment

iToBoS has bounding boxes only, no segmentation masks (dataset spec,
`docs/project_state.md` dataset roles table). There is no ground truth
to compute Dice against on TBP images, and manually annotating a mask
set is out of scope here (it would itself need a scoped decision if the
proxy signals below indicate a real problem — see decision rule).
Proxy metrics + structured visual audit are the only measurable
alternative, same reasoning as the geometry spec's "honest, measurable
alternative" framing.

## Data

Use CV-2's **real B1 detections on real iToBoS images** — not
simulated boxes. Source: `evaluation/cv2/prediction_diagnostics/b1_1280/predictions.csv`
(has `image_id, x1, y1, x2, y2, confidence, matched, zero_lesion`),
joined against `data/raw/itobos/_train/_train` (or `_test/images`) and
`data/splits/itobos_detection/val.csv`.

- Filter to `matched == True` (true-positive detections only — a
  false-positive box isn't a real lesion, so CV-3's output on it isn't
  informative about domain transfer, it's informative about CV-2, which
  is already characterized).
- Filter to `zero_lesion == False`.
- Convert pixel `x1,y1,x2,y2` → normalized YOLO xywh, run through the
  actual `src/inference/crop_normalize.py::crop_and_normalize` at the
  now-confirmed `margin=0.25` default (no `center_offset_frac` — that
  parameter was a geometry-experiment-only simulation knob, not a real
  input).
- Run CV-3 (`checkpoints/cv3_512/best.pt`) on every resulting crop.

## Proxy metrics (no ground truth required)

Compute all of these on the iToBoS crop predictions AND, as a control,
on the ISIC test-set predictions already produced for the baseline
(`evaluation/cv3/per_image_metrics.csv`) — the control matters because
these proxies have no absolute pass/fail value on their own, only
relative to CV-3's known-good behavior on its training domain:

1. **Degenerate-mask rate** — fraction with 0 foreground pixels or
   >95% foreground. A jump vs. the ISIC control means CV-3 is failing
   to find any coherent boundary on TBP images.
2. **Mask-to-crop area ratio** — distribution (median, IQR) compared
   to the ISIC control. A large shift signals systematic
   over-/under-segmentation on TBP inputs (plausible causes: hair,
   different scale/distance, lighting).
3. **Border-touching rate** — fraction of predicted masks with
   foreground pixels on the crop edge. Compared to ISIC control; a
   large jump suggests either lesions are being cut off or CV-3 is
   producing degenerate edge-hugging masks on the new domain.

These are sanity nets, not the decision signal — a metric shift without
visual confirmation is not actionable on its own.

## Visual audit (the actual decision signal)

Stratified random sample, **n=50**, stratified by iToBoS `sun_damage_level`
and `body_part` (same metadata fields already used in
`scripts/audit_cv2_pathological_images.py`, so the stratification code
is reusable) to avoid an accidentally easy or accidentally hard sample.

Produce a contact sheet (crop + predicted mask overlay), same format as
`evaluation/cv3/contact_sheet.jpg`, and manually rate each as
**reasonable** (mask plausibly bounds a lesion-like region) or **fail**
(empty, full-frame, spatially unrelated to any visible lesion, or
obviously wrong shape). One rater, one pass — this is a bounded
proxy-audit, not an inter-rater-reliability study.

## Pre-committed acceptance criteria (set BEFORE running)

- **Pass (domain-tolerant):** ≥ 80% of the n=50 sample rated
  "reasonable," AND no proxy metric shows a qualitatively different
  distribution shape (not just a numeric shift — e.g. a bimodal
  degenerate-vs-fine split) vs. the ISIC control. → CV-3 is usable
  as-is on TBP crops. Wire CV-2→CV-3→CV-4 for end-to-end pipeline work.
  Document the domain caveat but do not block on it.
- **Fail (domain gap is real):** < 80% reasonable, or proxies show a
  qualitative failure pattern (e.g. degenerate-mask rate >> control).
  → Do NOT immediately start collecting TBP masks or fine-tuning — first
  distinguish, from the failure_cases contact sheet itself, whether this
  is (a) a *scale/framing* problem (TBP crops are farther-away/smaller
  lesions than dermoscopic training images — possibly fixable by
  adjusting `margin` again, a one-line retry) vs. (b) a genuine
  appearance-domain problem (hair, lighting, skin texture CV-3 has never
  seen — not fixable by margin, needs real TBP training signal). Only
  (b) justifies opening a mask-collection/fine-tuning scoped follow-up,
  and that follow-up gets its own separate spec at that time, not here.

## Anti-rabbit-hole boundary

One sample (n=50), one visual pass, one set of proxy metrics computed
once (not swept). Apply the decision rule above and stop. This is not
an invitation to enlarge the sample chasing a rounder number, add more
proxy metrics, or start fine-tuning speculatively. Same discipline as
CV-2 Experiments D/E and the geometry interface spec.
