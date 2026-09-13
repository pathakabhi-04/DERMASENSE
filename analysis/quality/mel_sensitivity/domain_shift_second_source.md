# Second-source check: the referral head's collapse is general, the native classifier's is not

**Date:** 2026-09-13
**Script:** `scripts/run_domain_shift_check.py` (consolidates the DDI-specific
scripts into a reusable `scripts/evaluate_domain_shift.py`)
**Data:** DDI + DDI-2 (1,317 images, 25 melanomas, Stanford AIMI) plus a new
third source, **Fitzpatrick17k** (583 images, 83 melanomas), sampled from
`atlasdermatologico.com.br` -- a dermatology teaching atlas, independent of
Stanford/Redivis entirely. See "Acquisition note" below for why only one of
Fitzpatrick17k's two source domains was used.

## Why this was run

`native_classifier_on_ddi.md` found the domain-shift collapse on DDI/DDI-2
affected both CV-4b (AUC 0.9050 -> 0.5742) and the native 6-class
classifier (melanoma routing 0.7180 -> 0.3200), and flagged one open
question before treating that as general: is it a property of clinical
photography, or an artifact of one Stanford-sourced dataset? This is that
check.

## Result: the answer is different for the two scorers

| Source | native AUC | native MEL routed | head AUC | head MEL routed |
|---|---:|---:|---:|---:|
| ISIC2019 test (in-domain) | -- | 0.7180 | 0.9050 | 0.9024 |
| DDI (n=656, 21 MEL) | 0.6735 | 0.3333 | 0.5900 | 0.7619 |
| DDI-2 (n=661, 4 MEL) | 0.6537 | 0.2500 | 0.6120 | 0.7500 |
| **Fitzpatrick17k (n=583, 83 MEL)** | **0.7394** | **0.6867** | 0.6924 | 0.7952 |
| Combined (n=1900, 108 MEL) | 0.7341 | 0.6019 | 0.5982 | 0.7870 |

**The referral head's collapse is robust across all three independent
sources.** AUC sits at 0.59-0.69 on every one of them, well below the
0.9050 in-domain figure, regardless of institution or curation style. This
generalizes: CV-4b, specifically, does not transfer off ISIC dermoscopy to
any clinical-photo source tested so far.

**The native classifier's collapse does NOT generalize the same way.** It
is severe on DDI/DDI-2 (melanoma routing 0.25-0.33, a near-total collapse
from 0.72 in-domain) but much milder on Fitzpatrick17k (0.6867, close to
the in-domain figure). This directly contradicts the single-source
conclusion in `native_classifier_on_ddi.md` that the collapse "implicates
the shared representation" broadly -- with a second source, the more
accurate statement is that **the native classifier's failure is
source-dependent**, and DDI/DDI-2's images specifically are much harder
for it than Fitzpatrick17k's.

## What might explain the source-dependence, and what does not

Checked and ruled out: **resolution/file size is not the explanation.**
Fitzpatrick17k images (sampled: 353-526px wide, 33-61KB) are systematically
*smaller and more compressed* than DDI's (sampled: 555-1377px wide,
176-412KB) -- if raw image quality drove the difference, DDI should
transfer better, not worse. It does the opposite.

**Tested directly in a follow-up audit and NOT confirmed** (see
`domain_composition_audit.md`): the hypothesis was that Fitzpatrick17k's
atlas images are curated teaching examples -- selected, likely cropped
and framed by the atlas's own editors to show a lesion clearly -- while
DDI/DDI-2 are less-curated real clinical patient photography, and that
this framing difference explained the better transfer. Measured directly
using CV-3's own segmentation: DDI actually has the *largest* lesion-to-
frame area ratio of the four sources compared (including the model's own
training-domain reference), and Fitzpatrick has the *highest*
mask-touches-border rate -- both the opposite of what this hypothesis
predicted. The explanation for the native classifier's source-dependence
remains open, and is not blocking for the CV-4b work below, which is a
separate, already-settled finding.

## What this changes about the retraining/adaptation scoping question

The three-source picture argues for **treating the two problems
separately, not as one "domain shift" fix**:

1. **CV-4b (the referral head) has a general, reproducible transfer
   failure.** Refitting/retraining it on a broader feature distribution
   that includes non-ISIC images is justified independent of which
   clinical-photo source is used to validate it -- the collapse shows up
   everywhere tested.
2. **The native classifier's failure is NOT shown to be general.** Before
   committing to backbone-level retraining on the strength of this
   evidence, the open question is *why* DDI/DDI-2 specifically are harder
   -- capture/framing convention is the leading unconfirmed hypothesis
   above. That is a cheaper, more targeted investigation (a visual/
   composition audit, or testing on a curated-vs-uncurated split within
   one source) than committing to a full retraining program based on
   "clinical photography in general is a problem," which this check does
   not actually support for the native classifier.

## Caveats

- 108 combined melanomas is still a directional sample, not a validated
  deployment metric; per-source melanoma counts range from 4 (DDI-2) to
  83 (Fitzpatrick17k).
- Fitzpatrick17k's `three_partition_label=malignant` spans a wide range of
  malignancies (SCC, BCC, melanoma, mycosis fungoides, Kaposi sarcoma),
  matching how DDI's own `malignant` boolean is used identically in
  `referral_head_on_ddi.md` -- methodologically consistent across sources,
  not a new inconsistency introduced here.
- Per-image results: `analysis/quality/mel_sensitivity/domain_shift_per_image.csv`.
  Per-source/combined summary: `domain_shift_summary.csv`.

## Acquisition note: Fitzpatrick17k's larger source domain is dead

The full public CSV (16,577 rows) draws images from two atlas sites.
`dermaamin.com` (12,631 rows, 76% of the dataset) returns HTTP 200 on its
site root but 404 on every one of 5 tested original image paths -- the
`/clinical-pic/...` URL structure this CSV was captured against no longer
exists on that site. `atlasdermatologico.com.br` (3,905 rows) is
unaffected: 6/6 test URLs resolved, and the full sampled acquisition
(583 images) succeeded at 100%. `scripts/acquire_fitzpatrick17k.py`
restricts to the live domain rather than spending time on confirmed-dead
links.
