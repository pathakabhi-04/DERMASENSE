# The domain-shift problem is not confined to the referral head

**Date:** 2026-09-13
**Script:** `scripts/evaluate_native_classifier_on_ddi.py`
**Data:** same DDI + DDI-2 set as `referral_head_on_ddi.md` (1,317 images, 25 melanomas)
**Purpose:** scope the phone/clinical-photo domain-shift finding before any
retraining decision -- does it affect only CV-4b's linear referral probe
(fit exclusively on ISIC features), or the shared, fully-supervised
6-way classifier too (which WAS fine-tuned on PAD-UFES, i.e. on real
smartphone/clinical images, unlike the referral head)?

## Result: both fail, in different and worse ways than either looked alone

| | AUC (DDI+DDI2) | melanoma routed (DDI+DDI2) | melanoma routed (in-domain) |
|---|---:|---:|---:|
| Native 6-way classifier (shipped product path) | 0.6597 | **0.3200 (8/25)** | 0.7180 |
| Referral head (CV-4b) | 0.5742 | 0.7600 (19/25) | 0.9024 |

The in-domain "0.7180" figure is confirmed reproducible in this session by
re-running `scripts/evaluate_pad_checkpoint_on_isic_mel.py` directly
(0.7177 on 666 ISIC melanomas -- matches to rounding).

**Neither number is good, and they fail in opposite directions:**

- The **native classifier under-refers**. Its overall referral rate on
  this data is 46.8% (616/1317) -- not degenerate, not saturated, looks
  like a reasonably calibrated decision boundary in isolation. But its
  melanoma-specific routing **collapses to 0.32, worse than its own
  in-domain 0.72**, meaning the boundary is calibrated for the wrong
  part of the clinical-photo feature space: it is confidently and
  specifically wrong about which lesions are the malignant ones, not
  just cautious across the board.
- The **referral head over-refers**. Its overall referral rate is 81.8%,
  near-saturated regardless of true label (`referral_head_on_ddi.md`).
  Its higher melanoma-routing figure (0.76) is largely an artifact of
  referring almost everything, not genuine discrimination -- confirmed
  by its near-chance AUC (0.5742).

**AUC is the more informative number here** because it is threshold-independent:
the native classifier's 0.6597 vs the referral head's 0.5742 shows the
fully-supervised 6-way head retains *somewhat* more cross-domain structure
than the ISIC-only linear probe -- consistent with the native head having
actually seen PAD-UFES (real clinical/smartphone) images during its own
fine-tuning, where the referral head never saw anything but ISIC-domain
features. But 0.66 is still far short of a working classifier, and it did
not translate into better melanoma routing -- the opposite happened.

## What this settles, and what it doesn't

**Settles:** this is not simply "the referral head overfit to a narrow
ISIC-only feature distribution, refit it and move on." The shared
backbone/classifier -- fine-tuned end-to-end on PAD-UFES, which the
project already classifies as real smartphone/clinical photography --
still produces materially worse and specifically miscalibrated melanoma
routing on a *different* clinical-photo source (DDI/DDI-2) than it does
in its own training domain. That is evidence the domain gap is broader
than "dermoscopy vs. everything else": PAD-UFES's own capture conditions
(whatever clinical protocol produced it) do not appear to generalize even
to other clinical, non-dermoscopic photography, let alone to uncontrolled
consumer phone photos -- which is a further, still-larger, still
unmeasured step beyond what this analysis covers.

**Does not settle:** *why*. Candidate contributors -- none tested here,
none preferred over another: capture device and resolution, compression,
white balance/color calibration, framing/distance conventions, patient
positioning, DDI's own skin-tone distribution against PAD-UFES's. Any of
these could be primary; disentangling them is a separate, later
investigation, not concluded by this measurement.

## What this means for the retraining/adaptation decision

**Retraining or refitting the referral head alone is not indicated as
the priority.** Its ISIC-only fitting is A contributor to its own
collapse, but the more fully-trained native classifier -- which does not
share that specific weakness -- still fails on this domain, and fails
specifically on the metric that matters (melanoma routing, not just raw
AUC). That points at the shared representation (the backbone, and
plausibly the fine-tuning distribution it was exposed to) as a
higher-leverage target than a narrow linear-probe refit, consistent with
`docs/cv_metrics_improvement_plan.md` Step 2's broader retraining
direction -- now with a concrete, measured symptom (melanoma routing
0.72 -> 0.32 across a domain shift within "clinical photography" itself)
rather than only the previously-known dermoscopy-vs-clinical gap.

**Before spending retraining effort on anything:** the open, cheaper
question is whether this specific failure is DDI/DDI-2-idiosyncratic
(a property of these particular 1,317 images -- lighting, source
institution, camera type) or representative of clinical photography
broadly. A second, independent clinical-photo source would settle that
at the same low cost this measurement already demonstrated, before
committing to any backbone-level fix informed by data from a single
external source.

## Caveats

- 25 melanomas total; the melanoma-routing figures carry wide intervals
  (native: 95% CI [0.17, 0.52]; referral head: [0.57, 0.89] per
  `referral_head_on_ddi.md`) and should be read directionally.
- `HIGH_RISK_DIAGNOSES` (`src/risk/action_mapping.py`) sums softmax mass
  over {MEL, BCC, SCC} for the AUC computation; "referred" for the
  routing figures uses the shipped `predicted_class not in {NEV, SEK}`
  rule, identical to `SEES_CLINICIAN` in
  `scripts/evaluate_pad_checkpoint_on_isic_mel.py`.
- Per-image results: `analysis/quality/mel_sensitivity/native_classifier_on_ddi.csv`.
