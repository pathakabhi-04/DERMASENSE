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

**Re-entry gate — RE-SCOPED 2026-09-01.** The original wording is kept
below for the record, but it was unsatisfiable as written and did not
need to be satisfiable.

*Original:* "When the end-to-end pipeline (through CV-8) is built, the
mandatory end-to-end evaluation must include an explicit measurement of
how much CV-2's ~19% complete-miss rate degrades the final risk output.
IF that evaluation shows CV-2 miss rate is a dominant contributor to
end-to-end failure, THEN run the tiling experiment."

*Why it was unsatisfiable:* CV-2 runs only on the wide-field branch, and
iToBoS carries no diagnosis labels — confirmed, its splits hold only
`body_part`, `sun_damage_level`, `pixel_spacing`. There is no ground
truth to score a final risk decision against on that branch. The only
branch with diagnosis labels (PAD-UFES, pre-framed) skips CV-2 entirely
by design. So "how much does CV-2's miss rate degrade the final risk
output" cannot be measured with available data.

*Why it never needed to be:* the cost of a CV-2 miss is **structural,
not diagnostic**. A lesion CV-2 does not detect is never segmented,
classified, or risk-assessed — it is invisible to the entire downstream
pipeline. That cost is already measured (~19% of lesion-containing
images surface nothing) and requires no end-to-end risk score.

*Re-scoped gate:* the real question is whether that miss rate is
acceptable given the product's deployment model, which is a product
judgement, not a measurement. Current answer: **tiling stays deferred on
priority grounds, and is effectively closed.** The intended primary
input is a zoomed-in photo of a lesion the user is already concerned
about (pre-framed branch); wide frames occur incidentally. Two
consequences:

1. Wide-field is not the primary path, so a TBP-imagery miss rate is not
   a primary-path product risk.
2. More decisively, an incidental wide frame from a user is a **phone**
   photo, not TBP-rig imagery. CV-2 is trained and validated exclusively
   on iToBoS TBP images, so its real-world number on wide-field phone
   photos is *unmeasured*. Tiling would tune the detector against a
   domain the product may never see.

**Therefore: do not run tiling.** The operative open question is the
already-tracked "CV-2 domain validation (iToBoS→phone)" deferred item,
which must be answered before any further CV-2 refinement is
justifiable. Re-open tiling only if the deployment model changes to make
wide-field a primary input path AND the phone-domain gap is closed
first.

End-to-end assembly measurements that informed this:
`analysis/product_eval/cv1_cv4_assembly/result.md`.

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