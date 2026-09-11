# The model with worse accuracy is the safer one

**Date:** 2026-09-12
**Sample:** the same 666 ISIC2019-test melanomas for both checkpoints.
**Script:** `scripts/evaluate_pad_checkpoint_on_isic_mel.py`

## Result

| | ISIC-trained (8-class, **in-domain**) | PAD-transferred (6-class, **cross-domain**) |
|---|---:|---:|
| MEL recall | **0.5661** | 0.4505 |
| Sent to a clinician | 64.0% | **71.8%** |
| **Told "low risk" (MONITOR)** | 35.4% (236) | **28.2% (188)** |

The deployed PAD-transferred checkpoint is **11 points worse at
recognising melanoma** and **7 points better at doing the right thing
about it**.

## Why

The two models fail in different directions.

```
ISIC-trained, of 666 melanomas:   NV 167 + BKL  69 = 236 -> benign -> MONITOR
PAD-transferred, of 666:          NEV 164 + SEK  24 = 188 -> benign -> MONITOR
                                  BCC  82 + SCC  55 = 137 -> malignant -> URGENT
```

Both miss melanoma at similar rates. The ISIC model's misses land on
benign classes; the PAD model's land substantially on *other
malignancies*. A melanoma called BCC still produces
`URGENT_EVALUATION` and the patient sees someone. A melanoma called
nevus produces `MONITOR` and the patient is reassured.

PAD-UFES is 36% BCC and 2.3% MEL, so fine-tuning on it biases the model
toward malignant predictions. That bias is accidental — nobody designed
it — and it is worth more clinically than the 11 points of recall it
cost.

The PAD model achieves this **while handicapped by domain shift**
(trained on clinical photographs, evaluated here on dermoscopy), which
should hurt it. It routes better anyway.

## What this proves

**Class accuracy is the wrong objective, demonstrated rather than
argued.** Ranking these two checkpoints by MEL recall or Macro-F1 picks
the one that reassures more melanoma patients. Every training decision
so far has optimised a metric that is actively misaligned with the
product's only safety-critical outcome.

This is the empirical case for the change already proposed in
`docs/cv_metrics_improvement_plan.md` Step 2 — now with evidence:

1. **Optimise the melanoma→benign boundary**, not overall accuracy. The
   188 matter; the 137 mostly do not.
2. **Make "routed to a clinician" the headline metric.** It ranks these
   checkpoints correctly and MEL recall does not.
3. **A malignant/benign head is likely to beat both.** The product asks
   "does this need a doctor?", and both models answer that better than
   they answer "which of six is it?".

## Caveats

- Cross-domain for the PAD model, in-domain for the ISIC model. The
  comparison is not like-for-like, and the domain handicap makes the
  PAD result a *lower* bound — the finding survives the caveat rather
  than depending on it.
- PAD-UFES's own test split has 9 melanomas, so in-domain measurement
  of the deployed model remains impossible at useful precision. That is
  a data-collection problem, not an analysis one.
- 28.2% is still 188 melanoma patients told "low risk". Better is not
  acceptable.
