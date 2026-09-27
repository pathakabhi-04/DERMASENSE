# Academic Track — ViT Backbone on CV-3 and CV-4, with Real Attention Maps

**Status:** SPEC — not yet run. It depends on Experiment 2
(`docs/academic_resnet50_three_task_spec.md`), whose ResNet-50 CV-3 result is
this spec's segmentation comparator.
**Companion to:** Experiment 1 (shared encoder) and Experiment 2 (same
ResNet-50 backbone, trained separately). This experiment changes the backbone
family and nothing else.

---

## 1. Question

1. **Performance.** If ResNet-50 is replaced by a ViT-B/16 backbone, with the
   data, split, head, loss and schedule unchanged, does each task come within
   a fixed margin of its ResNet-50 counterpart, or beat it?
2. **Attention.** A ViT has real attention weights. Do its attention maps
   concentrate on the lesion, measured with the same ground truth, metrics,
   sanity check and claim rule as Experiment 2's Grad-CAM maps? And how do
   they compare with Grad-CAM computed on the same ViT?

---

## 2. Why this experiment, and why only CV-3 and CV-4

- Experiment 2 had to use Grad-CAM saliency as the "attention map",
  because a convolutional network has no attention weights (Exp 2 §5.1). A
  ViT does, so this is the only way the academic write-up can report real
  attention maps.
- **Only the backbone changes.** The classification head (pooled feature →
  linear), the U-Net decoder, the losses, the data and the metrics are kept
  exactly as in Experiment 2. A difference in results can then be attributed
  to the backbone family.
- **CV-2 detection is out** (Section 4). iToBoS lesions are about 30 px at
  1280 px, roughly two 16×16 patches. Global attention over the resulting 6,400
  tokens is expensive, and a hierarchical detector (ViTDet, Swin) is a
  different design, not a backbone swap. It gets its own spec if this one ends
  in outcome A (Section 7).
- **The ViT does not revive the joint model.** A ViT backbone still needs
  boxes, masks and diagnoses on the same images for a single joint model.
  Experiment 1's outcome C stands.

---

## 3. Arms

| Task | ViT model | Data / split | Comparator (ResNet-50) |
|---|---|---|---|
| **CV-4 classification** | `torchvision` `vit_b_16`, `ViT_B_16_Weights.IMAGENET1K_V1`; head: CLS token → Linear(768, 8) | ISIC 2019, existing CV-4 split (seed 42), 224×224 | Production CV-4 weighted ResNet-50: test macro-F1 **0.5756** |
| **CV-3 segmentation** | same ViT-B/16 as the encoder; simple feature pyramid (Li et al. 2022, ViTDet) from the last block to strides 2 / 4 / 8 / 16 / 32 with ResNet-50's channel counts (64 / 256 / 512 / 1024 / 2048), feeding the **unchanged** U-Net decoder from `src/academic/model.py` | ISIC 2018 Task 1, existing CV-3 split, 512×512 (position embeddings interpolated from 14×14 to 32×32) | Experiment 2 ResNet-50 U-Net: test Dice (from `evaluation/academic_r50/performance.csv`) |

**Pretraining is held fixed at ImageNet-1k.** ResNet-50 used ImageNet-1k
weights, and `IMAGENET1K_V1` is the ImageNet-1k ViT-B/16. Stronger ViT
pretraining (SWAG, DINOv2, ImageNet-21k) would mix the backbone change with a
pretraining-data change, so it is out of scope (Section 4). Known limitation:
ViTs pretrained on ImageNet-1k alone are weaker than on larger corpora, so a
ViT shortfall here is a result about this pairing, not about ViTs in general.

**Capacity is not matched.** ViT-B/16 has about 86M parameters against
ResNet-50's 25M. This is reported as a limitation and not corrected (no ViT-S
in torchvision; changing depth would be a second change).

### 3.1 Training settings (fixed now, before any run)

Each task keeps its ResNet-50 counterpart's recipe, with **one** ViT-specific
change applied identically to both tasks: the backbone learning rate is
3e-5 (heads and decoder keep 1e-4), with linear warmup over the first epoch.
AdamW at 1e-4 without warmup is a known failure mode when fine-tuning ViTs.
This is the only deviation, and it is not swept.

| Setting | CV-4 (ViT) | CV-3 (ViT) |
|---|---|---|
| Recipe it copies | CV-4 weighted (`configs/cv_resnet50_weighted.yaml` on main) | Experiment 2 CV-3 (Exp 2 §3.1) |
| Optimizer | AdamW, weight decay 1e-4 | AdamW, weight decay 0.01 |
| LR | backbone 3e-5, head 1e-4, 1-epoch linear warmup, then constant | backbone 3e-5, pyramid + decoder 1e-4, 1-epoch linear warmup, then constant |
| Batch / epochs | 32 / 10 | 8 / 50 |
| Loss | sqrt-inverse-frequency weighted CE | 0.5 BCE + 0.5 Dice |
| Augmentation | CV-4 (flips, ±15°, colour jitter) | none |
| Selection | best val macro-F1 | best val Dice |
| Precision | fp16 autocast | fp16 autocast |
| Seed | 42 | 42 |

**Data:** ISIC 2019 train/val/test goes into a new bundle, pre-resized to
224×224 (the CV-4 eval resize). The CV-3 and HAM data reuse
`academic_r50_bundle` and `academic_joint_bundle`, which are already on the
volume.

---

## 4. Scope

**In:** ViT-B/16 backbone for CV-3 and CV-4, performance against ResNet-50,
and attention maps.

**Out, and why:**
- **CV-2 detection with a ViT** (Section 2). It gets its own spec, and only
  after outcome A.
- **Other ViTs or pretraining** (ViT-S, Swin, DeiT, SWAG, DINOv2): each is a
  second change.
- **Learning-rate or schedule sweeps.** Section 3.1 is fixed.
- **A joint ViT model.** Experiment 1 outcome C.

---

## 5. Performance decision rule (committed before running)

This is single-seed, like the Experiment 2 comparators. The margins come from
the seed-to-seed spread measured in Experiment 1: classification macro-F1 SD
≈ 0.02, segmentation Dice SD ≈ 0.008 on ISIC 2018 test.

| Task | "Matches" if | "Beats" if | Metric source |
|---|---|---|---|
| CV-4 | test macro-F1 ≥ 0.5756 − 0.02 = **0.5556** | ≥ 0.5756 + 0.02 = **0.5956** | same test split and metric as CV-4 |
| CV-3 | test Dice ≥ R50 Dice − **0.01** | ≥ R50 Dice + **0.01** | `scripts/academic_r50/evaluate_cv3.py` |

| Outcome | Condition | Action |
|---|---|---|
| **A** | Both match | Claim: a ViT backbone serves segmentation and classification as well as ResNet-50 (report "beats" separately). Unlocks a ViT detection spec. |
| **B** | Exactly one matches | Finding: the swap works for one task only. Report which. |
| **C** | Neither matches | Finding: an ImageNet-1k ViT-B/16 underperforms ResNet-50 on these datasets at these sizes. |

"Beats" is only reported. With a single seed it is marked as such, and it
does not change the outcome letter.

---

## 6. Attention maps

### 6.1 Two maps per ViT model

| Map | Method | CV-4 target | CV-3 target |
|---|---|---|---|
| **Attention (primary)** | Attention rollout (Abnar & Zuidema, 2020): per block, heads averaged, residual added as 0.5·A + 0.5·I, rows renormalised, multiplied through all 12 blocks | row of the CLS token, over the 14×14 patches | mean of the rows of the patch tokens inside the predicted mask (prob ≥ 0.5), over the 32×32 patches |
| **Grad-CAM (bridge)** | as in Experiment 2, on the last block's patch tokens reshaped to the patch grid | predicted-class logit | sum of mask logits inside the predicted mask |

The Grad-CAM map is the bridge to Experiment 2: ViT Grad-CAM against ResNet-50
Grad-CAM compares backbones with the method held fixed, and ViT attention
against ViT Grad-CAM compares methods with the model held fixed.

### 6.2 Scoring, sanity check, claim rule (identical to Experiment 2)

- Evaluation sets, ground truth and metrics are as in Exp 2 §5.3: HAM test
  (1,451) with Tschandl masks for CV-4, and ISIC 2018 test (260) for CV-3.
  Metrics are pointing game, energy-in-lesion and IoU@0.5, all at
  model-input resolution.
- Sanity check (Adebayo et al., 2018): on the same 50-image sample rule,
  re-initialise the **last transformer block** and recompute each map. If the
  median Spearman correlation is ≥ 0.5, that map fails, and no lesion-focus
  claim may be made from it.
- A claim of the form "the ViT attends to the lesion" requires the sanity
  check to pass **and** pointing game ≥ 0.80 **and** energy-in-lesion ≥ 0.50,
  judged per map.
- Known limitation, stated up front: attention weights are not guaranteed to
  be explanations (Jain & Wallace, 2019; Serrano & Smith, 2019). Passing the
  claim rule shows that attention concentrates on the lesion, not that the
  attention causes the prediction.
- Figures: one contact sheet per task and map, using the Exp 2 selection rule
  (seed 42; CV-4: 10 correct + 10 misclassified). Each ViT sheet uses the same
  images as the matching Experiment 2 sheet, so the two can be compared side
  by side.

---

## 7. Preconditions (stop if any fails)

1. Experiment 2 is complete, and its ResNet-50 CV-3 test Dice is committed.
2. `vit_b_16(IMAGENET1K_V1)` loaded in this code reproduces the published
   ImageNet-1k top-1 (81.07%) on 1,000 ImageNet val images within ±1.5 points.
   If ImageNet val is unavailable, check instead that the logits match
   torchvision's own model on 10 random tensors (max abs difference < 1e-4).
3. Smoke test: one CV-3 batch at 512 with the interpolated position
   embeddings runs forward and backward, and memory is recorded.

---

## 8. Anti-rabbit-hole boundary

After the two training runs and the map evaluation, apply Sections 5 and 6.2
and stop. Do **not**:
- change the learning rate, warmup, epochs or augmentation if the ViT misses;
- swap pretraining (SWAG, DINOv2, 21k) or architecture (Swin, DeiT, ViT-S);
- add seeds to move a borderline result;
- try other attention readouts (raw last-layer attention, Chefer et al. LRP,
  attention flow) to get cleaner-looking maps;
- investigate single images where a map looks odd.

---

## 9. Deliverables

- `evaluation/academic_vit/performance.csv`: ViT vs ResNet-50 per task, with
  the Section 5 thresholds.
- `evaluation/academic_vit/attention_metrics.csv`: per task and map
  (attention, Grad-CAM), the §6.2 metrics and sanity correlation, with
  Experiment 2's ResNet-50 Grad-CAM rows repeated for comparison.
- `evaluation/academic_vit/attention_{cv3,cv4}_{rollout,gradcam}.jpg`: contact
  sheets.
- `docs/academic_vit_result.md`: outcome letter, claims allowed or withheld,
  limitations.
