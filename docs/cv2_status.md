# CV-2 Component Status

**Status:** BASELINE ACCEPTED AT FLOOR — refinement deferred (scheduled, not abandoned).

**Date frozen:** end of Experiment E session.

---

## What "baseline accepted at floor" means

CV-2 is a working, documented baseline suitable to build the rest of the CV
pipeline on. It is NOT marked "done" — one identified refinement lever
remains, deferred to a gated future decision. This distinction is
deliberate: the risk of "build now, refine later" is that the baseline
silently becomes permanent. The gate below prevents that.

## Baseline performance (val split, conf=0.25, revised Section 22 metrics)

| Metric | Value | Gate | Met? |
|---|---|---|---|
| Image-level recall | 0.8098 | >= 0.90 | No — at accepted floor |
| Zero-lesion FP burden median | 0.00 | <= 1 | Yes |
| Zero-lesion FP burden p90 | 1.00 | <= 2 | Yes |

Checkpoint: YOLO11s, `runs/cv2/e_yolo11s-2/weights/best.pt` (on network
volume). Note: E (YOLO11s) and B1 (YOLO11n) perform within noise of each
other on image-level recall; the smaller B1 checkpoint is an equally valid
baseline if a lighter model is preferred for the pipeline. Either is
acceptable — they are functionally equivalent on the gates.

## Why 0.81 is an acceptable floor (not a failure)

CV-2 is a candidate localizer feeding CV-3/CV-4 and, ultimately, a
human-in-the-loop risk engine (CV-8). Image-level recall of 0.81 means it
surfaces >= 1 true lesion in 81% of lesion-containing images. The pipeline
is explicitly designed so that absence of a CV-2 detection does not equal
absence of disease — the risk engine and human review are the safety net for
the ~19% miss. For an MVP, this is a defensible operating point.

## The deferred refinement, and its re-entry gate

**Untried lever:** tiling / SAHI inference — slicing wide-field images into
overlapping tiles so small lesions become relatively larger per-tile,
directly attacking the extreme small-object scale (median lesion 0.00097 of
image area) that is the suspected root cause of the recall ceiling.

**Why deferred rather than run now:** Four experiments (threshold sweep,
resolution, oversampling, capacity) confirmed the ceiling is not reachable
by simpler levers. Tiling is the one mechanistically-motivated remaining
option but carries real cost (inference-time slicing, cross-tile box
merging, and it changes the CV-2 -> CV-3 crop interface). That cost is only
justified if CV-2's miss rate turns out to be a dominant pipeline failure —
which cannot be known until the full pipeline exists.

**Re-entry gate (specific, not "someday"):** When the end-to-end pipeline
(through CV-8) is built, the mandatory end-to-end evaluation must include an
explicit measurement of how much CV-2's ~19% complete-miss rate degrades the
final risk output. IF that evaluation shows CV-2 miss rate is a dominant
contributor to end-to-end failure, THEN run the tiling experiment (one run,
per the standing stopping rule). If it is not dominant, CV-2 stays at floor
and the tiling lever is formally closed.

## Constraints on building downstream components on this baseline

1. **Do not develop CV-3 on CV-2's detected crops.** CV-3 (segmentation)
   should continue to be developed and evaluated on clean ground-truth
   crops (ISIC 2018), so CV-3's own quality is measurable independent of
   CV-2's 19% miss. The compounding effect is surfaced separately in the
   CV-2 -> CV-3 interface validation (already on the roadmap).

2. **Downstream stages must treat "CV-2 surfaced nothing" as a known mode,**
   not assume CV-2 is complete. The architecture already accommodates this
   (risk engine + human-in-the-loop), which is part of why building forward
   on this baseline is reasonable.

3. **The CV-2 -> CV-3 interface validation is where this baseline gets its
   real-world test** — deferred to when it is most informative (once CV-3 is
   solid), not skipped.

## Evidence trail
- `experiment_d_result.md` — oversampling negative result.
- `experiment_e_result.md` — capacity negative result + full 4-experiment record.
- `cv2_section22_finalized.md` — revised metrics, committed gates, stopping rule.
- This file — the accepted-baseline decision and deferred-refinement gate.