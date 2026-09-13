# Compositional audit: the leading hypothesis is not supported

**Date:** 2026-09-13
**Script:** `scripts/audit_domain_composition.py`
**Data:** DDI (656), DDI-2 (665), Fitzpatrick17k (583), and PAD-UFES's own
test split (352, as the reference domain the shipped model's "whole
photo is the crop" convention was built around).

## Why this was run

`domain_shift_second_source.md` found the native classifier's collapse is
severe on DDI/DDI-2 (melanoma routing 0.72 → 0.25–0.33) but mild on
Fitzpatrick17k (0.69), and offered one unconfirmed hypothesis: Fitzpatrick's
atlas images are curated teaching examples, possibly cropped/framed by
editors closer to how ISIC/PAD-UFES training images were captured, while
DDI/DDI-2 are less-curated real patient photography. This audit tests that
directly, reusing CV-3 (already-deployed segmentation, not new tooling) to
measure composition rather than assume it.

## Result: the hypothesis is not supported — two measurements contradict it outright

| source | n | degenerate mask | touches border | area fraction (median) | blur (median) | contrast (median) |
|---|---:|---:|---:|---:|---:|---:|
| DDI | 656 | 0.8% | 35.5% | **0.360** | 0.75 | 0.80 |
| DDI-2 | 665 | 1.8% | 19.7% | 0.249 | 0.46 | 1.00 |
| Fitzpatrick17k | 583 | 1.2% | **41.9%** | 0.305 | 1.00 | 0.71 |
| PAD-UFES (reference) | 352 | 2.6% | 23.9% | 0.312 | 0.52 | 0.45 |

(`blur`/`contrast` are `src/quality/signals.py` scores, clipped to [0,1]
against a fixed reference — higher is better/sharper; a median of 1.0
means at least half the images cleared the reference, not that they are
unboundedly sharper than one another beyond it.)

**Lesion-to-frame area ratio contradicts the hypothesis.** DDI has the
*largest* median area fraction of all four sources (0.360) — larger than
Fitzpatrick (0.305) and larger than the model's own training-domain
reference (0.312). If DDI images were wider/less lesion-centered than
Fitzpatrick's curated atlas shots, this number would run the other way.
It does not.

**Mask-touches-border contradicts it too.** Fitzpatrick has the *highest*
border-touching rate of any source (41.9%, versus DDI's 35.5%) — the
opposite of what "atlas images are more consistently cropped to the
lesion" would predict.

**CV-3 segmentation reliability does not explain it either.** DDI's
degenerate-mask rate (0.8%) is the *lowest* of all four sources, lower
than the model's own reference domain (2.6%). CV-3 is not struggling to
find a lesion in DDI images more than elsewhere — if anything less.

**Blur/contrast show real differences, but no consistent story.** DDI's
scores (0.75 blur, 0.80 contrast) are reasonably strong on their own
terms — not obviously worse than Fitzpatrick's. PAD-UFES's own reference
domain has the *lowest* contrast score of all four sources (0.4485) yet
is what the shipped model was built and calibrated against, which rules
out "low contrast alone predicts collapse" as a clean explanation too.

## What this settles

**The leading hypothesis from the second-source check is ruled out, not
confirmed.** Framing/composition, as measured by lesion area fraction,
border-touching, and CV-3's own segmentation reliability, does not
distinguish DDI/DDI-2 (severe native-classifier collapse) from
Fitzpatrick17k (mild collapse) in the direction the hypothesis predicted
— on two of the four measured axes it points the opposite way.

This is a genuine negative result, reported as one rather than stretched
into a confirming narrative. It means the real explanation for why the
native classifier fails much harder on DDI/DDI-2 specifically remains
open, and is not one of: zoom/framing, segmentation reliability, or (on
the evidence here) blur/contrast in any simple way.

## What is not ruled out, and is not investigated here

Color/skin-tone distribution, JPEG compression signatures, camera/sensor
characteristics, white-balance and lighting-temperature differences,
and DDI's specific patient population versus Fitzpatrick's atlas source
population are all still open candidates. None is tested in this audit.
Per the same bounded-experiment discipline this whole investigation has
followed: this was one cheap, pre-scoped check with a clear negative
result, not license to keep widening the search on momentum. Any further
compositional investigation is a new, separately-scoped check, not a
continuation of this one.

## Why this does not block the CV-4b work

CV-4b's (the referral head's) collapse is a *separate, already-settled*
finding: it is general across all three independent sources regardless
of composition (`domain_shift_second_source.md`). Whatever explains the
native classifier's source-dependence, it does not change that the
referral head needs a broader-than-ISIC-only refit — that conclusion
does not depend on resolving this open question.

## Caveats

- Median values only in the table above; full distributions in
  `analysis/quality/mel_sensitivity/domain_composition_audit.csv`.
- "PAD-UFES (reference)" uses the model's own test split, not a
  from-scratch relabelled sample — it represents the domain the shipped
  checkpoint's conventions were built around, used here as a baseline for
  what "normal" composition looks like to this pipeline, not an
  independent validation set.
