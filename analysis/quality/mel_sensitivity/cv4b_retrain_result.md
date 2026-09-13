# CV-4b refit result: partial recovery, decision rule not met, checkpoint unchanged

**Date:** 2026-09-13
**Script:** `scripts/refit_referral_head_broad.py`
**Plan:** `cv4b_retrain_scope.md` (read that first — this reports against
the rule it pre-committed, not a rule chosen after seeing the number)

## Correction (2026-09-13, same day): this split leaks patients

Found while designing the backbone fine-tune
(`docs/cv4b_backbone_finetune_design.md` §4), and recorded here rather
than quietly regenerating the numbers.

`cv4b_retrain_split.csv` splits DDI-2 **by image**, but DDI-2 has repeat
patients — 89 patients contribute 204 of its 661 images. **25 of the 132
held-out DDI-2 test images below come from patients that also appear in
the training portion.** Two photos of the same patient are not
independent samples.

The bias runs optimistic, so the conclusion drawn here — *the bar was
not met* — is not overturned by it; the true non-ISIC figure is if
anything slightly worse than the 0.7333 reported. But 0.7333 is a
contaminated number and should be cited as such. DDI-2 was also the
weakest source in the table (AUC 0.6211), which bounds how much the leak
could have flattered the combined figure.

`scripts/build_cv4b_splits.py` supersedes this split with a
patient-grouped one that asserts no group and no image spans two splits.
Everything below is left as originally written and measured.

## Result

| | AUC | melanoma routed | benign referred |
|---|---:|---:|---:|
| ISIC test, **before** refit (ISIC-only fit) | 0.9050 | 0.9024 | 0.3940 |
| ISIC test, **after** refit (broad fit) | 0.9027 | 0.8769 | 0.3642 |
| non-ISIC test, **before** refit (baseline collapse) | 0.5982 | 0.7870 | -- |
| non-ISIC test, **after** refit (broad fit, held out) | **0.7333** | **0.8333** | **0.5418** |

Per-source, held out (never seen during fit or threshold selection):

| source | n | AUC | melanoma routed | n_mel |
|---|---:|---:|---:|---:|
| DDI | 131 | 0.6273 | 0.6667 [0.35, 0.88] | 9 |
| DDI-2 | 132 | 0.6211 | 1.0000 [0.21, 1.00] | 1 |
| Fitzpatrick17k | 117 | 0.7795 | 0.9000 [0.70, 0.97] | 20 |

## Against the pre-committed decision rule

- ISIC test AUC ≥ 0.85: **met** (0.9027 — essentially unchanged, no
  collapse from broadening the training distribution).
- non-ISIC test AUC ≥ 0.80: **not met** (0.7333, up from the 0.5982
  combined baseline, but short of the bar).

**Per the rule set in advance, the refit head is not written to
`checkpoints/referral_head/referral_head.json` — it is unchanged, still
the ISIC-only fit.** Confirmed by `git status` showing no diff on the
checkpoint after running this.

## Reading the result honestly: real, partial, not sufficient

This is not a null result. AUC moved from 0.60 to 0.73 combined, and
every one of DDI-2's single held-out melanoma, and 9/10 of
Fitzpatrick's, were routed. That is training-data-driven recovery, not
noise — the pre-committed bar was deliberately set above the "did
anything move" question, at a level meant to represent something closer
to usable.

It is also not free. The 40%-benign-referral operating point was chosen
on the *combined* validation pool, and it does not spend evenly:
**non-ISIC benign referral came out at 54.2%**, well above the 36.4% it
now costs on ISIC (was 39.4% pre-refit) and above the flat 40% budget
the threshold was nominally chosen for. A broader fit made the
melanoma-routing number look better partly by referring non-ISIC benign
lesions much more aggressively — a real cost that a single global
threshold obscures.

## What this does and doesn't settle, per the scope doc's own rule

The scope doc pre-committed: if the bar isn't met, treat that as
evidence pointing at the frozen backbone's representation, not license
to keep tuning this head. Two things are true at once here:

1. **Some of the gap is a data problem.** More diverse training data for
   the linear probe recovered real ground (0.60 → 0.73 AUC), which a
   pure representation ceiling would not allow.
2. **A meaningful gap remains at 0.80 target, and it isn't free once
   there** — the 54% non-ISIC benign-referral rate suggests the probe is
   compensating for weak separation by referring broadly, not by
   actually discriminating better on non-ISIC images.

Consistent with the scope doc's rule: this is not escalated into a
hyperparameter search on this same linear probe (more regularization
sweeps, different C, etc. chasing 0.80). The residual gap is best read
as a soft representational ceiling — the frozen ISIC-trained backbone's
2048-d features encode ISIC-relevant structure well and non-ISIC
structure only partially, and no amount of refitting the *linear* layer
on top changes what the backbone extracted in the first place.

## What this changes about next steps

- **Do not ship the broad-fit head as-is.** It doesn't clear its own
  bar, and its benign-referral cost on non-ISIC data (54%) is worse than
  advertised by looking at AUC alone.
- **Backbone-level adaptation (fine-tuning, not just a new linear
  probe) is the next lever**, and it needs the GPU budget this project
  has explicitly deferred. This result is the evidence that justifies
  raising that as a real, costed decision rather than assuming it.
- **The ISIC-only head currently shipped stays in place** — it is not
  worse than the broad-fit alternative on the one thing it was validated
  for (ISIC, 0.9050/0.9024), and the broad-fit alternative both costs
  more per non-ISIC benign lesion and still misses its own bar.

## Caveats

- Split: 60/20/20, stratified by (source, is_malignant), seed 20260913,
  written to `cv4b_retrain_split.csv` before fitting.
- DDI-2's held-out test slice has exactly 1 melanoma — the 1.0000
  routing figure is n=1 and not a rate in any meaningful sense; reported
  with its Wilson interval ([0.21, 1.00]) rather than as a clean number.
- Label harmonization caveat from the scope doc applies unchanged: non-ISIC
  "malignant" is a coarser label than ISIC's referral-need `REFER` set.
