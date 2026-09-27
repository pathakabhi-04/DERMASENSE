# Academic Track — ViT Backbone on CV-3 and CV-4, with Real Attention Maps

**Status:** SPEC — not yet run. Revised 2026-09-28, before any run, after a
review of main's CV-2/3/4 docs (see §2.1). Experiment 2 is complete; its
ResNet-50 CV-3 test Dice, **0.8964**, is this spec's segmentation comparator.
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
  different design, not a backbone swap. It is **not** unlocked by any
  outcome here: main's CV-2 stopping rule (`docs/cv2_section22_finalized.md`,
  `docs/cv2_status.md`) closes further detector work until the iToBoS→phone
  domain gap is answered, and a new wide-field detector architecture is
  gated on a product decision (`docs/build_on_baseline_1.md` §B item 1).
- **The ViT does not revive the joint model.** A ViT backbone still needs
  boxes, masks and diagnoses on the same images for a single joint model.
  Experiment 1's outcome C stands.

### 2.1 What main already settled, and what that means here

Reviewed 2026-09-28 so that this experiment does not repeat settled work:

- **Capacity is not what limits CV-4.** ResNet-18 and ResNet-50 gave the same
  melanoma recall (0.56 vs 0.54–0.57; `docs/cv_metrics_improvement_plan.md`
  on main). A larger or different backbone is therefore **not expected to
  improve CV-4**, and this experiment is not a CV-4 improvement attempt. Its
  value is the backbone comparison and the real attention maps. A CV-4 "beats"
  result would be reported under this prior, not as a product lever.
- **The unseen-source gap is a data constraint** (`docs/data_constraint_spec.md`,
  four experiments). Nothing here is evaluated on, or may be claimed for, an
  unseen capture source such as a user's phone.
- **ISIC 2019 pre-resizing was already chosen by measurement**
  (`docs/cv4b_backbone_finetune_design.md` §11). 256 px bilinear and JPEG q95
  were rejected on paired feature fidelity; **320 px LANCZOS q100** was
  validated (feature cosine 0.993 against originals). This spec reuses that
  setting and bundle (§3.1), instead of the new 224 px bundle an earlier
  draft proposed.
- **Higher CV-3 resolution was already tried** (`checkpoints/cv3_768` on main:
  val Dice 0.829 at 768 vs 0.865 at 512). CV-3 stays at 512.
- **Grad-CAM++, Score-CAM and other CAM variants are ruled out**
  (`docs/cv5_explainability_spec.md`), consistent with §8.
- **CV-3 comparisons use a paired bootstrap** over the 260 test images
  (`docs/cv3_segmentation_baseline.md` §7). §5 adds it.

---

## 3. Arms

| Task | ViT model | Data / split | Comparator (ResNet-50) |
|---|---|---|---|
| **CV-4 classification** | `torchvision` `vit_b_16`, `ViT_B_16_Weights.IMAGENET1K_V1`; head: CLS token → Linear(768, 8) | ISIC 2019, existing CV-4 split (seed 42), 224×224 input from the validated 320 px bundle (§3.1) | Production CV-4 weighted ResNet-50, **re-scored on the same bundle images** (R50ᵦ, §5). Originals: test macro-F1 0.5756 (context) |
| **CV-3 segmentation** | same ViT-B/16 as the encoder; simple feature pyramid (Li et al. 2022, ViTDet) from the last block to strides 2 / 4 / 8 / 16 / 32 with ResNet-50's channel counts (64 / 256 / 512 / 1024 / 2048), feeding the **unchanged** U-Net decoder from `src/academic/model.py` | ISIC 2018 Task 1, existing CV-3 split, 512×512 (position embeddings interpolated from 14×14 to 32×32) | Experiment 2 ResNet-50 U-Net: test Dice **0.8964** (`evaluation/academic_r50/performance.csv`) |

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

**Data:**
- **ISIC 2019: reuse main's validated `cv4b_bundle`**, at 320 px LANCZOS
  q100 JPEG. It covers 18,062 / 3,304 / 3,473 of the 18,402 / 3,375 / 3,554
  train/val/test images. **The gap is exactly DF and VASC**: that bundle held
  only the ISIC classes that map onto PAD-UFES's six, and DF and VASC have
  none. The 492 missing images (train 340, val 71, test 81) are added with
  the **same** function (`resize_one` in `scripts/preresize_cv4b_dataset.py`
  on main), so all eight classes share one preprocessing. At train and eval
  time the images are resized to 224×224 with the CV-4 transform, as main
  did with this bundle.
- **Upload it as a single tar**, not loose files: per-object overhead
  dominates at this file count (main §12.4, and the slow mask upload in
  Experiment 1).
- CV-3 and HAM data reuse `academic_r50_bundle` and `academic_joint_bundle`,
  which are already on the volume.

---

## 4. Scope

**In:** ViT-B/16 backbone for CV-3 and CV-4, performance against ResNet-50,
and attention maps.

**Out, and why:**
- **CV-2 detection with a ViT** (Section 2). Not unlocked by any outcome
  here. Main's CV-2 stopping rule applies.
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
| CV-4 | test macro-F1 ≥ R50ᵦ − 0.02 | ≥ R50ᵦ + 0.02 | CV-4 test split (3,554), both models scored on the **same** bundle images |
| CV-3 | test Dice ≥ 0.8964 − 0.01 = **0.8864** | ≥ 0.8964 + 0.01 = **0.9064** | `scripts/academic_r50/evaluate_cv3.py` |

**R50ᵦ** is the production CV-4 ResNet-50 scored on the bundle images, so
the two backbones see identical inputs. It must come within ±0.01 of 0.5756
(its score on the originals, reproduced in Experiment 2). If it does not,
the pre-resized images are not faithful enough for this 8-class task, which
main validated only for its binary referral head: stop and report before
training.

**R50ᵦ measured 2026-09-28, before any ViT run**
(`evaluation/academic_vit/r50_on_bundle.json`,
`scripts/academic_vit/score_r50_on_bundle.py`):
- Macro-F1 **0.5683**, which is 0.0073 below 0.5756, so the ±0.01 gate
  **passes**.
- The CV-4 thresholds are therefore **match ≥ 0.5483** and **beats ≥ 0.5883**.
- Recorded as a finding: on the bundle, melanoma recall is **0.533**
  [0.495, 0.571] (355/666), against 0.566 (377/666) on the originals. The
  pre-resize costs about 22 melanomas while macro-F1 moves only 0.007. The
  comparison stays fair, because both models see identical inputs, but the
  preprocessing is not neutral for melanoma, and main validated it only for a
  binary head.

**Reported alongside, not decision-changing:**
- CV-3: a paired bootstrap (10,000 resamples, 95% CI) of ViT − ResNet-50
  Dice over the 260 test images (`scripts/academic_r50/paired_bootstrap_cv3.py`).
- CV-4: melanoma recall on the 3,554-image test split, with a Wilson
  interval, for both models. Main treats this as the metric that matters
  (`docs/cv_metrics_improvement_plan.md`), and macro-F1 alone can hide a
  melanoma regression.

| Outcome | Condition | Action |
|---|---|---|
| **A** | Both match | Claim: a ViT backbone serves segmentation and classification as well as ResNet-50 (report "beats" separately). Unlocks nothing further; detection stays closed (§2). |
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
4. The completed ISIC 2019 bundle has all 25,331 split images (18,402 / 3,375
   / 3,554), checksums verify, and R50ᵦ is within ±0.01 of 0.5756 (§5).

**Status 2026-09-28 (local):**
- Precondition 1: passed (Experiment 2 done).
- Precondition 2: passed on CPU (`evaluation/academic_vit/preflight_cpu.json`).
  Our trunk equals torchvision's with a max difference of 0.0. The hand-run
  attention blocks used for rollout reproduce the model's own forward
  exactly (0.0 at 224 and at 512). Rollout rows sum to 1.
- Precondition 3: passed on CPU. The 512 px ViT segmenter (101.8M
  parameters) runs forward and backward. The GPU memory figure is recorded
  by `run_all.sh`'s preflight on the pod.
- Precondition 4: passed. The bundle has all 25,331 images (24,839 from
  cv4b; 492 DF/VASC resized with main's function, which reproduced 50/50
  cv4b files byte for byte), and R50ᵦ is 0.5683.
- Attention inputs: the ViT CV-4 maps are computed on the same HAM test
  originals as Experiment 2's ResNet-50 maps, so the two map sets share
  inputs.

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
  the Section 5 thresholds, R50ᵦ, melanoma recall with Wilson intervals, and
  the CV-3 paired-bootstrap CI.
- `evaluation/academic_vit/attention_metrics.csv`: per task and map
  (attention, Grad-CAM), the §6.2 metrics and sanity correlation, with
  Experiment 2's ResNet-50 Grad-CAM rows repeated for comparison.
- `evaluation/academic_vit/attention_{cv3,cv4}_{rollout,gradcam}.jpg`: contact
  sheets.
- `docs/academic_vit_result.md`: outcome letter, claims allowed or withheld,
  limitations.
