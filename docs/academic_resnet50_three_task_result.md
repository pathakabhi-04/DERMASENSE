# Academic Track — One Backbone, Three Tasks (ResNet-50) + Attention Maps: Result

**Spec:** `docs/academic_resnet50_three_task_spec.md` (settings in §3.1, fixed before the first run)
**Runs:** CV-3 ResNet-50 U-Net (seed 42, 50 epochs, 0.29 h) and CV-2 Faster R-CNN (seed 42, 50 epochs, 4.71 h), completed 2026-09-27 on RunPod. CV-4 reused, no training.
**Code:** `src/academic/`, `scripts/academic_r50/` at `50cc28b`
**Raw numbers:** `evaluation/academic_r50/{performance.csv, attention_metrics.csv, decision.json, sanity_*.csv}`
**Figures:** `evaluation/academic_r50/attention_{cv2,cv3,cv4}.jpg`

---

## Outcome: **A — one backbone serves all three tasks at production level**

Both new models meet their Section 4 thresholds. The detection result is
marginal on false positives (Section 1.1), and that is part of the finding.

## 1. Performance

| Task | Metric | ResNet-50 | Production baseline | Threshold | Pass |
|---|---|---|---|---|---|
| CV-2 | Image-level recall (val, conf 0.25) | **0.8304** | 0.8098 (YOLO11s E) | ≥ 0.78 | yes |
| CV-2 | Zero-lesion FP burden, median | 0 | 0 | ≤ 1 | yes |
| CV-2 | Zero-lesion FP burden, p90 | **2** | 1 | ≤ 2 | yes, **at the limit** |
| CV-3 | Test Dice (ISIC 2018, 260) | **0.8964** | 0.8640 (U-Net from scratch) | ≥ 0.854 | yes |
| CV-4 | Test macro-F1 (ISIC 2019) | 0.5756 | 0.5756 | by construction | yes |

Secondary numbers, reported only (they do not change the letter):

| Task | Metric | ResNet-50 | Baseline |
|---|---|---|---|
| CV-2 | Box-level recall | 0.5422 | 0.5023 |
| CV-2 | Binary zero-lesion FPR (share of lesion-free images with ≥ 1 false candidate) | **0.4726** | 0.1758 |
| CV-2 | Burden median / p90 over **all 349** zero-lesion images (spec §3.1 correction) | 0 / 1.2 | 0 / 1.0 |
| CV-3 | Test IoU | 0.8295 | 0.7851 |
| CV-3 | Test median Dice | 0.9313 | 0.9136 |

### 1.1 Reading the performance result

- **CV-3: a clear improvement.** The ImageNet-pretrained ResNet-50 encoder
  under the unchanged U-Net recipe beats the from-scratch U-Net by +0.032 Dice
  and +0.044 IoU on the same test split. This is one seed. Experiment 1 put the
  seed-to-seed spread on this test set at about 0.008 Dice, so the gap is
  about four times that spread. Best epoch 41 of 50 (val Dice 0.888).
- **CV-2: recall slightly higher, false alarms clearly worse.** Faster R-CNN
  catches a lesion in 83.0% of lesion-containing images against YOLO11s's
  81.0%, and it finds more individual boxes (0.54 vs 0.50). But it puts at
  least one false candidate on **47%** of lesion-free images, against 18% for
  YOLO. The spec's burden gate still passes, since median 0 and p90 exactly 2,
  but with no headroom. The spec's recall ceiling for this dataset (0.81,
  `docs/cv2_status.md` on main) is not broken either: 0.83 is below the 0.90
  "notable" line.
- **Consequence for the product pipeline:** none is made from this
  experiment. Outcome A says one backbone type *can* serve all three tasks;
  it does not say ResNet-50 should replace YOLO11s for CV-2. On false alarms,
  YOLO11s remains the better detector.
- **CV-4:** loaded by key remapping, the production checkpoint reproduced
  the committed test confusion matrix except for one near-tie image (top-2
  logit gap 0.0012; macro-F1 0.575535 vs 0.575637). This was accepted as
  reproduced.

## 2. Attention maps (Grad-CAM; "attention map" as defined in spec §5.1)

| Task | n | Pointing game | Energy in lesion | IoU@0.5 | Sanity: median Spearman (orig vs randomised) | Sanity | Claim allowed |
|---|---|---|---|---|---|---|---|
| CV-4 | 1,451 HAM test images | **0.966** | **0.726** | 0.315 | 0.194 | pass | **yes** |
| CV-3 | 259 ISIC 2018 test images (1 excluded: empty prediction, `ISIC_0013208`) | **0.988** | **0.799** | 0.259 | 0.204 | pass | **yes** |
| CV-2 | 3,190 true-positive boxes, iToBoS val | 0.020 | 0.002 | — | 0.008 | pass | **no** |

Claim rule (§5.3): sanity pass, pointing game ≥ 0.80, and energy-in-lesion
≥ 0.50.

### 2.1 Claims allowed

- **"The CV-4 classifier attends to the lesion."** Its Grad-CAM maximum falls
  inside the dermatologist mask on 96.6% of HAM test images, and 72.6% of map
  energy lies inside the mask. The maps are tied to learned weights: after
  layer4 is randomised, the rank correlation with the original maps drops to a
  median of 0.19.
- **"The CV-3 segmenter attends to the lesion."** It has pointing 98.8% and
  energy 79.9%, and its maps likewise collapse when layer4 is randomised
  (median ρ 0.20). The CV-3 target is the model's own predicted mask, so this
  claim is expected. It confirms that the method behaves, rather than
  revealing something surprising.

### 2.2 Claim withheld

- **CV-2: no lesion-focus claim.** The map's maximum lands inside the matched
  ground-truth box on only 2.0% of true positives, and the box holds 0.2% of
  the map's energy. The contact sheet shows why: with standard Grad-CAM
  (spatially averaged gradients) at layer2, the map lights up every
  lesion-like spot on the skin, not the one box being explained. The maps are
  still tied to the learned weights (median ρ 0.008 after randomisation), so
  this reflects the method, not noise. A box-local readout would be a
  different method, and §6 forbids switching methods to get better-looking
  maps. Reported as a limitation.

### 2.3 Other observations (from the figures, not investigated per §6)

- IoU@0.5 is low for CV-4 and CV-3 (0.32 and 0.26). The maps come from 7×7
  and 16×16 grids, so they locate the lesion but do not trace its outline.
  IoU is not part of the claim rule.
- On CV-4's misclassified examples, the map is still centred on the lesion.
  So the errors are not explained by the model looking away from the lesion.
  This is an observation from 10 images, not a measured result.

## 3. Limitations

1. **One seed per new model**, as the spec sets, so there are no confidence
   intervals. The CV-2 p90 burden sits exactly on its limit and could flip
   with another seed.
2. **Val is both the CV-2 report set and the baseline's report set.** Faster
   R-CNN used no val selection (final epoch), while YOLO11s E's best
   checkpoint was selected on val. If anything, that favours the baseline.
3. **The CV-2 burden figures inherit the baseline script's omission** of
   zero-lesion images with no candidates (§3.1). The corrected all-349
   figures are reported alongside and do not change the pass.
4. **Grad-CAM is saliency, not attention.** ResNet-50 has no attention
   weights. Real attention maps are Experiment 3 (`docs/academic_vit_spec.md`).
5. **Different pretraining corpora.** CV-2 starts from COCO detection
   weights, while CV-3 and CV-4 start from ImageNet. "One backbone" means the
   same architecture, not the same weights.
6. **The CV-3 comparison changes more than the backbone:** it swaps a
   from-scratch U-Net for a pretrained ResNet-50 encoder, and adds ImageNet
   normalisation, which the pretrained encoder requires.

## 4. What happens next

- Section 6 applies: no anchor tuning, tiling, detector swaps, CAM-layer or
  method changes, or extra seeds.
- **Experiment 3** (`docs/academic_vit_spec.md`) can now run, since its
  precondition 1 is met. Its CV-3 comparator is Dice **0.8964**, so the ViT
  "matches" if Dice ≥ 0.8864 and "beats" if ≥ 0.9064.
