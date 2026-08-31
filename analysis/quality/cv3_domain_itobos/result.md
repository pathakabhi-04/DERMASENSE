# CV-3 Domain Validation on iToBoS (TBP) Crops — Result

**Status:** Borderline FAIL against the pre-committed gate (39/50 = 78%,
gate was >= 80%). Failure pattern is dominated by mask fragmentation, not
scale, so this reads as a real (if partial) dermoscopic->TBP appearance
gap rather than a fixable crop-geometry issue. See decision below —
this does NOT trigger fine-tuning automatically; that stays a separate,
explicitly-scoped decision per the spec.

## Setup
Per `docs/cv3_domain_validation_spec.md`. Real CV-2 B1 true-positive
detections (matched==True) on real iToBoS images, through the actual
`crop_and_normalize()` at margin=0.25, through CV-3
(`checkpoints/cv3_512/best.pt`).

- Proxy-metric pass: 1000 iToBoS crops (bounded random subsample, seeded
  — see `DEFAULT_MAX_ITOBOS_CROPS` rationale in the script; full 5686
  would be ~60min on this CPU for a metric that isn't the decision
  signal) + 260 ISIC control crops (same code path, GT box + margin=0.25).
- Visual audit: stratified-by-sun_damage_level random sample, n=50,
  manually rated by inspection of `audit_contact_sheet.jpg` against
  `audit_sample.csv`.

## Proxy metrics (sanity net — not the decision signal)
| metric | iToBoS | ISIC control |
|---|---|---|
| degenerate-mask rate | 0.018 | 0.000 |
| border-touch rate | 0.026 | 0.062 |
| fg_frac median | 0.279 | 0.334 |
| fg_frac IQR | 0.195 | 0.112 |

No single proxy shows an alarming collapse (degenerate rate is low in
absolute terms, border-touch is actually lower on iToBoS). The widened
fg_frac IQR (0.195 vs 0.112) was the one flag — consistent with what the
visual audit then confirmed: more variable, sometimes-fragmented
predictions on iToBoS, not a uniform shift.

## Visual audit (the decision signal): 39/50 reasonable (78%)

Gate was >= 80% (40/50). Missed by one image.

**Failure breakdown (11 fails):**
- **Fragmented / scattered blobs** (6): image_1797, image_2727,
  image_2822, image_4519, image_5283, image_5457 — CV-3 predicts several
  small disconnected regions instead of one coherent lesion boundary.
  This is the dominant failure mode (55% of fails).
- **Near-empty / miss** (3): image_0878, image_3856, image_4311 — mask
  is empty or a single stray pixel-scale fragment.
- **Blocky border artifact** (2): image_8331, image_4231 — a
  rectilinear, straight-edged region touching the crop border, visually
  distinct from an organic lesion boundary.

**Scale/framing check** (first diagnostic step the spec's fail-branch
calls for, before considering fine-tuning): compared original CV-2 box
area (px²) for failed vs. reasonable crops.
- Failed: mean 1009 px², median 917 px²
- Reasonable: mean 1288 px², median 1153 px²

Failed crops skew slightly smaller but the distributions overlap heavily
(smallest crop overall, 287px², is a fail; the second-smallest, 342px²,
passed; several fails sit mid-range at 1800-2200px², well within the
"reasonable" range). **This is not a clean scale cutoff** — box size
does not predict failure. That rules out "just retry with a larger
margin" as a fix.

## Decision (per docs/cv3_domain_validation_spec.md)

Result is a FAIL (78% < 80%), and the diagnostic points at (b) — a
genuine appearance-domain gap (texture/hair/lighting CV-3 has not seen,
manifesting as fragmented predictions), not (a) a scale/framing issue
fixable by adjusting margin.

Per the spec, this is where the fail branch explicitly stops and does
NOT auto-launch fine-tuning: "Only (b) justifies opening a
mask-collection/fine-tuning scoped follow-up, and that follow-up gets
its own separate spec at that time, not here." That follow-up would need
real TBP mask signal, which doesn't exist yet (no iToBoS segmentation
ground truth) — collecting or weakly-generating it is a real scoping
decision (cost, method, how much) that deserves its own explicit
go/no-do, not an automatic next step from this experiment.

## What this means for the pipeline right now
- CV-3 is not silently broken on TBP images — the majority (78%) of
  real CV-2 detections still get a reasonable segmentation. This is
  usable for continued end-to-end pipeline wiring (CV-4 development,
  product-level eval) with the domain gap now measured and documented,
  not just caveated as "not yet evaluated."
- The ~22% fragmentation-dominated failure rate on TBP crops is now a
  known, quantified limitation, not an unknown. It should feed into
  CV-8/product-level thinking (e.g., downstream stages already have to
  treat "CV-2 surfaced nothing" as a known mode per `docs/cv2_status.md`
  — a fragmented/low-quality CV-3 mask on ~1 in 5 TBP images is the same
  category of "handle the imperfect case," not a blocking defect).
- Fine-tuning CV-3 on TBP data is a real, identified option for later,
  gated on an explicit decision to invest in TBP mask collection — not
  triggered automatically by this result.
