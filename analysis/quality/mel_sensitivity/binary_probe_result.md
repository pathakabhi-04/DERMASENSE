# The backbone already knows. The 6-class head is throwing it away.

**Date:** 2026-09-12 · **Script:** `scripts/probe_binary_separability.py`
**Cost:** no GPU, no training of any network. Seconds of logistic regression
on features cached during the SCC/BCC investigation.

## Question

78 of 666 ISIC-test melanomas receive **<5% total malignant probability**
from the deployed 6-class head — confidently wrong, and unrecoverable by
any threshold (`threshold_vs_retraining.md`).

Before spending GPU on a binary head: is that information even present in
the representation, or absent from it?

**Pre-committed rule:** a linear probe recovering ≥50% of the 78 at ≤40%
benign referral means the signal is present and a head is justified.
Below that, the signal is absent and the effort belongs in data.

## Result — the signal is present, and strongly

Linear probe (`LogisticRegression`, 5-fold out-of-fold) on the **frozen
2048-d penultimate features**, refer {MEL,BCC,SCC,AK} vs benign {NV,BKL}:

| benign referred | all melanomas recovered | **the 78** recovered |
|---:|---:|---:|
| 10% | 0.760 | 0.397 |
| 20% | 0.862 | 0.590 |
| **30%** | **0.907** | 0.744 |
| 40% | 0.952 | **0.885** |
| 50% | 0.976 | 0.949 |

Against what the 6-class head can do with the same backbone:

| approach | melanoma routed | benign referred |
|---|---:|---:|
| argmax (shipped) | 0.718 | 0.232 |
| best probability threshold | 0.806 | 0.380 |
| **linear probe** | **0.907** | **0.300** |
| **linear probe** | **0.952** | 0.400 |

**A linear probe on frozen features clears the ≥0.90 target that no
threshold on the 6-class head could reach** — and does it at a *lower*
benign referral rate than the best threshold managed for 0.806.

## What this means

1. **The backbone is not the problem.** It encodes malignant/benign
   separability that the 6-class softmax discards. Every prior negative
   result — class weighting, capacity, SupCon — was tuning the wrong
   thing: they all optimised the 6-way objective.
2. **Retraining the backbone is not indicated.** A head on frozen
   features is minutes of CPU, not GPU-days. That is the cheapest
   intervention with the largest measured effect in this project.
3. **It explains the PAD-vs-ISIC routing inversion**
   (`pad_vs_isic_routing.md`). If malignant/benign is linearly decodable,
   then a model whose errors scatter within "malignant" routes better
   than one whose errors cross into "benign" — which is exactly what was
   observed, and is a property of the *decision rule*, not the features.

## Caveats — stated, not buried

- **This is a decodability measurement, not a validated model.** The
  probe is trained out-of-fold on the ISIC *test* features. The feature
  extractor never saw ISIC test (it was fine-tuned on PAD-UFES), so
  there is no extractor leakage, but a shippable head must be trained on
  ISIC **train** features and evaluated on test untouched.
- **Still dermoscopy.** Every number here is ISIC. Phone photos remain
  unmeasured and will be worse.
- **AK counted as referral.** It routes to `EVALUATE_SOON`, so this
  measures "sent to a clinician", consistent with the rest of this
  analysis.
- The 30–40% benign referral rate is a real cost and a product decision.
  The shipped `MONITOR → REVIEW` gate already reviews 18.8%, so this is
  an increase, not a new category of burden.

## Next step, and it is small

Train the same probe properly: ISIC **train** features → evaluate on
ISIC test, untouched. If it holds, the head ships as an additional
referral signal alongside the 6-class output — the 6-class prediction
still drives narration, the binary head drives routing.

No backbone retraining. No GPU. This supersedes
`docs/cv_metrics_improvement_plan.md` Step 2.
