# The referral head does not transfer to clinical photography

**Date:** 2026-09-13
**Script:** `scripts/evaluate_referral_on_ddi.py`
**Data:** DDI + DDI-2 (Stanford AIMI, Redivis), 1,317 usable images
(656 + 661; 5 unreadable/missing), biopsy-proven, clinical (non-dermoscopic)
photography with diverse skin tones. Acquired via `scripts/acquire_ddi.py`.

## Why this was run

Every strong number for CV-4b (0.9024 melanoma routing @ 39.4% benign
referral, `referral_head_result.md`) is measured on ISIC2019 -- dermoscopy,
a dedicated instrument under controlled lighting. `docs/product_scope_and_readiness.md`
gate 2 names the dermoscopy-to-phone gap as unmeasured and the largest open
unknown in the project. DDI/DDI-2 are the closest available public stand-in:
biopsy-proven, ordinary-camera, diverse-skin-tone clinical photographs. This
is Step 1 of the phone-gap plan -- measure before touching any model.

**Pre-committed decision rule**, fixed before running:

```
melanoma routing >= 0.85  -> the head transfers, proceed to phone validation
0.70 <= routing < 0.85    -> domain adaptation justified, now with a metric
routing < 0.70            -> not phone-deployable yet, report plainly
```

## Result

| | melanoma routed | benign referred | malignant routed |
|---|---:|---:|---:|
| ISIC2019 test (dermoscopy, in-domain) | **0.9024** | 0.3940 | -- |
| DDI (n=656, 21 MEL) | 0.7619 (16/21) | 0.7608 | 0.8421 |
| DDI-2 (n=661, 4 MEL) | 0.7500 (3/4) | 0.8409 | 0.9571 |
| **Combined (n=1317, 25 MEL)** | **0.7600 (19/25)**, 95% CI [0.57, 0.89] | **0.8048** | -- |

**Overall referral rate, any label: 0.8178 (1077/1317).**

**Malignant-vs-benign AUC: 0.5742**, against 0.9050 on ISIC test. Confirmed by
re-scoring the ISIC test features through the identical code path
(`ReferralHead.decide()`) as a control: reproduces 0.9050 AUC / 0.9024
melanoma routing / 0.3940 benign referral to 4 decimal places, so the scoring
mechanism itself is not in question.

## Why the point estimate alone is misleading, and what actually happened

The 0.76 combined melanoma-routing figure sits in the pre-committed
"0.70-0.85, domain adaptation justified" band, which reads as a moderate,
survivable degradation. It is not that. The **overall referral rate is
82%**, near-identical to the malignant-routed rate (84-96% by dataset) --
the head is referring almost every image on this domain regardless of
label, not selectively catching malignancy. Combined with an AUC of 0.57
(chance is 0.50), the finding is that **the score has stopped
discriminating malignant from benign on clinical photography.** The
melanoma-routing number is largely an artifact of near-universal referral,
not a real, weaker version of the ISIC signal.

This distinction is the entire reason AUC was reported alongside the
operating-point pass rate. Two different failures were possible:

1. The score still ranks correctly, but the threshold (fit on ISIC's score
   distribution) sits on the wrong part of the range for this domain --
   fixable by recalibrating a threshold alone, no retraining.
2. The representation itself stops separating the classes -- needs
   retraining or domain adaptation, a threshold cannot fix it.

AUC 0.57 is (2), not (1). An illustrative domain-refit threshold (chosen
on this same eval set, so not a valid held-out estimate -- included only
to show the shape of the problem) does not rescue it either: near-chance
AUC means no threshold on this feature distribution meaningfully
separates the classes.

## What this does and does not indicate

- **Not a bug in this evaluation.** The backbone was loaded via the exact
  `NativeClassifierConfig(backbone="resnet50", pretrained=False,
  dropout=0.0)` + `strict=True` construction `NativePredictor.from_checkpoint`
  uses (src/inference/native.py) rather than an independent reconstruction,
  and the preprocessing transform matches `build_eval_transform` /
  `_base_preprocessing` (src/data/transforms.py) exactly: `Resize((224,224))`,
  `ToTensor()`, ImageNet normalization. Images loaded with plausible
  photographic pixel statistics (checked directly, not corrupted/blank).
  `ReferralHead.decide()` reproduces the documented ISIC numbers exactly
  when run through the identical code path on cached ISIC features.
- **A real, not-yet-explained domain-shift finding.** The frozen
  PAD-UFES/ISIC-trained backbone's 2048-d representation, which discriminates
  malignant from benign cleanly on dermoscopy (AUC 0.9050), does much closer
  to nothing on ordinary clinical photography (AUC 0.5742). Plausible
  contributors -- none confirmed, none tested here: no dermoscope-standard
  polarized lighting/oil interface, different color casts and white balance,
  JPEG compression artifacts, more visible surrounding skin/background,
  different capture distance and framing conventions. Disentangling these
  is future work, not concluded here.
- **DDI-2's 71.8%→95.7% malignant-routed range and its 4 melanomas make its
  own row very low-precision** -- reported for completeness, not as an
  independent confirmation. The combined 25-melanoma estimate (95% CI
  [0.57, 0.89]) is the number to treat as directional.

## Per-dataset detail

DDI is more contributory (21 of 25 melanomas) and covers a wider disease
vocabulary (see `data/raw/ddi/manifest.csv`); DDI-2 adds diverse-skin-tone
coverage (Asian patients specifically) at low melanoma count. Per-image
scores, labels, and referral decisions: `referral_head_on_ddi.csv`.

## What this changes about Step 1 of the phone-gap plan

The pre-committed rule said "0.70-0.85 -> domain adaptation justified,
now with a metric to target." That holds, but the AUC finding sharpens
the target: adaptation needs to recover ranking ability (AUC), not just
nudge a threshold. Retraining the referral head (or the shared backbone
representation it sits on) on a mix that includes clinical photography --
not further threshold-tuning on ISIC-only features -- is the indicated
next step, with AUC on a held-out clinical-photo set as the metric to
clear before any operating point is chosen.

**This result does not roll back CV-4b's contract wiring** (still correct
and validated on its own -- dermoscopy -- domain, still a strict floor on
`risk_category` that only ever prevents LOW, never asserts a degree of
risk). It is new evidence that the domain-shift caveat already stated in
`docs/cv8_contract_delta_v1.2.md` and `docs/product_scope_and_readiness.md`
is not a formality -- it is the single largest determinant of whether the
referral head means anything once it meets phone-camera input.
