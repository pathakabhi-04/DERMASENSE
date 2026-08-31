# CV-2 → CV-3 Interface Validation — Spec

**Status:** Committed before running (pre-registered criteria).

## What this validates
CV-3 (segmentation, UNet, ~0.86 Dice) was trained on whole ISIC 2018
images squash-resized to 512×512. In the real pipeline it will instead
receive CROPS from CV-2 detection. This experiment tests whether CV-3
degrades when its input is a tighter / more off-center crop than the
full lesion-centric frames it trained on.

## What this deliberately does NOT validate
- **Domain shift** (iToBoS TBP vs ISIC dermoscopic). CV-2's real crops
  come from iToBoS, which has no segmentation masks, so Dice cannot be
  computed on them. This experiment isolates crop GEOMETRY on ISIC (which
  has masks). The domain axis is a separate, later question.
- This is why "run CV-3 on CV-2's real crops and compute Dice" is not the
  experiment — it is not possible with available ground truth. Simulating
  the geometry shift on ISIC is the honest, measurable alternative.

## Method
- ISIC 2018 test set, 260 dermoscopic images with masks.
- Derive tight lesion box from each ground-truth mask.
- Sweep crop geometry: margin ∈ {1.0, 0.5, 0.25, 0.1, 0.0} (context
  expansion per side), center_offset ∈ {0.0, 0.1, 0.2} (localization
  error). 15 (margin, offset) cells.
- Run CV-3 on each crop (preprocessed identically to training via
  src/inference/crop_normalize.py), compute Dice (identical definition
  to baseline, imported from src.segmentation.metrics) vs the
  identically-cropped mask.

## Harness self-check (must pass before trusting results)
The **margin=1.0, offset=0.0** cell approximates CV-3's full-frame
training distribution (a generous crop around the centered lesion). It
must recover Dice close to the ~0.86 baseline (say, >= 0.80). If it does
NOT, the harness is measuring something wrong (preprocessing mismatch,
mask-crop misalignment) and results are invalid until fixed. Do not
interpret the tight-margin rows until this self-check passes.

## Pre-committed acceptance criteria (set BEFORE running)

The interface is considered **robust** if:
- At a realistic operating crop (margin=0.25, offset=0.1 — a moderately
  expanded, slightly-off-center crop, representative of a decent CV-2
  detection), CV-3 Dice stays **>= 0.75** (i.e. degradation from the
  ~0.86 whole-frame baseline of no more than ~0.11).

The interface is considered **fragile / needs work** if:
- Dice at margin=0.25/offset=0.1 falls **< 0.75**, OR
- Dice collapses steeply across the realistic crop range (e.g. tight
  crops < 0.5), indicating CV-3 cannot handle detector-style inputs
  without intervention.

## Decisions by outcome (committed)
- **Robust (>= 0.75 at realistic crop):** the interface works. Adopt the
  crop-normalization margin that maximizes Dice as the pipeline default.
  CV-2 → CV-3 join is validated (for geometry; domain still caveated).
  Move on — do NOT sweep dozens of margin values chasing marginal Dice.
- **Fragile (< 0.75):** the interface is a real problem. The fix options,
  in order of preference, are: (a) tune crop margin if a better margin
  exists in the grid; (b) if no margin works, CV-3 needs fine-tuning on
  detector-style crops — a scoped follow-up, not open-ended. Pick the
  indicated fix, apply once, re-validate once, then stop.

## Anti-rabbit-hole boundary
This is one bounded experiment answering one question: does CV-3 tolerate
CV-2-style crop geometry? It is not an invitation to exhaustively
characterize CV-3's crop sensitivity. Run the grid once, apply the
committed decision rule, move on. Same discipline as CV-2 Experiments D/E.