# Academic Track — One Backbone, Three Tasks (ResNet-50) + Attention Maps

**Status:** DONE — outcome **A** (CV-2 marginal on false positives). See `docs/academic_resnet50_three_task_result.md`.
**Companion to:** `docs/academic_joint_seg_cls_spec.md` (that spec shares one
encoder *across* tasks; this one trains the *same backbone* separately for
each task).

---

## 1. Question

1. **Performance.** If the same ResNet-50 backbone (ImageNet-pretrained) is
   trained separately for each of CV-2 (detection), CV-3 (segmentation) and
   CV-4 (classification), does each model come within a fixed margin of the
   current production model for that task?
2. **Attention.** For each of the three models, does the attention map
   concentrate on the lesion, measured against ground-truth masks and boxes?

---

## 2. Why one backbone, trained separately

A vision model is a **backbone** (image → feature maps, task-agnostic) plus a
**head** and loss that define the output (boxes, a mask for every pixel, or a
single set of class scores). The production pipeline uses three different
backbones (YOLO11s, a U-Net trained from scratch, ResNet-50), so differences
between tasks mix up the backbone and the head. Fixing the backbone at
ResNet-50 leaves only the head and the task changing. ResNet-50 is chosen
because CV-4 already uses it and because the joint spec's arms use it. It gives
one backbone across all academic experiments.

This is the "trained separately" half of the academic story. The joint spec is
the "shared" half. Detection is included here because training separately
avoids every blocker in joint-spec Section 2.1: each task keeps its own
dataset and labels.

---

## 3. Arms

| Task | Model | Data / split | Training | Production baseline (test / val as committed) |
|---|---|---|---|---|
| **CV-2 detection** | `torchvision` `fasterrcnn_resnet50_fpn_v2`, COCO-pretrained, 2 classes (bg, lesion) | iToBoS, `data/splits/itobos_detection/{train,val}.txt` (same as B1/E) | `min_size`/`max_size` set so the long side is 1280 (matches B1/E `imgsz`); 50 epochs; seed 42; other settings torchvision defaults | YOLO11s E: image-level recall **0.8098**, zero-lesion FP burden median **0** / p90 **1** (val, conf 0.25) |
| **CV-3 segmentation** | U-Net decoder on a `torchvision` ResNet-50 encoder (ImageNet), built in-repo (no new dependency) | ISIC 2018 Task 1, existing CV-3 split (259 val / 260 test) | 512×512; 0.5 BCE + 0.5 Dice; 50 epochs; seed 42 (all as the CV-3 baseline) | U-Net from scratch: test Dice **0.8640**, IoU 0.7851 |
| **CV-4 classification** | Existing weighted ResNet-50 checkpoint; **no retraining** | ISIC 2019 | — | test macro-F1 **0.5756** |

CV-4 is already ResNet-50, so it is the reference point and costs nothing.
The new training runs are CV-2 and CV-3: **two runs, one seed each**, matching
how their baselines were produced.

Implementation note: the CV-3 ResNet-50 U-Net is the same module the joint
spec's segmentation-only and joint arms need. Build it once and reuse it.

### 3.1 Settings (fixed 2026-09-27, before the first run)

The table in Section 3 names the model, data, resolution, loss, epochs and
seed. The remaining settings are fixed here, before any run.

**CV-2: Faster R-CNN.** "torchvision defaults" means the defaults of the
torchvision detection reference recipe (`references/detection/train.py`),
scaled to one GPU:

| Setting | Value |
|---|---|
| Weights | `FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`; box predictor replaced by `FastRCNNPredictor(1024, 2)`; default trainable backbone layers (3) |
| Resolution | `min_size=1280, max_size=1280`. iToBoS images are ≈1094×894 or 945×771, so the long side becomes 1280 (as §3 requires) |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4, lr 0.005 (the reference 0.02 at 16 images/batch, scaled linearly to batch 4) |
| Schedule | linear warmup over the first 1,000 iterations (factor 0.001); MultiStepLR ×0.1 at epochs 31 and 42 (the reference 16/22 of 26, scaled to 50) |
| Augmentation | horizontal flip 0.5 (the reference `hflip` policy) |
| Data | all 6,778 train images, including the 1,401 zero-lesion images (empty targets), as B1/E |
| Precision | fp16 autocast (the reference `--amp` option), for cost |
| Checkpoint | final epoch. The reference recipe does no val selection, and val is also the evaluation set |
| Export | `box_score_thresh=0.001`, NMS 0.5, `detections_per_img=100` (torchvision defaults apart from the score threshold, which §4 sets) |
| Matching | IoU ≥ 0.5, greedy IoU-first, exactly as `scripts/analyze_cv2_predictions.py` on main |

Baseline note: `predictions.csv` only has rows for images with at least one
candidate. E's committed burden figures therefore cover 330 of the 349
zero-lesion val images; the other 19 had no candidates at conf 0.001.
Recall is unaffected, since all 1,346 lesion images are present. The
decision uses the spec's script unchanged for both models. A corrected
burden over all 349 zero-lesion images is reported alongside it for both
models, as a secondary figure.

**CV-3: ResNet-50 U-Net.** This is the module from
`src/academic/model.py` (arm S), trained with the CV-3 baseline's settings:
batch 8, AdamW lr 1e-4, weight decay 0.01 (the PyTorch default the baseline
used), no augmentation, fp16, 50 epochs, seed 42, best by val Dice. The one
difference from the baseline is ImageNet mean/std normalisation, which a
pretrained encoder needs. Images and masks are stored already resized to
512×512 with the resize `evaluate_cv3.py` applies (cv2 linear / nearest).

> **Correction (2026-09-28, after the run, recorded rather than rewritten):**
> "batch 8" came from `docs/cv3_segmentation_baseline.md` and
> `train_cv3.py`'s default. The official baseline checkpoint's own
> `config.json` (`checkpoints/cv3_512`) records batch 16. The run used 8
> as written above. See the result doc, limitation 7.

**CV-4.** `isic2019_resnet50_weighted_best.pt`, loaded into a torchvision
ResNet-50 by key remapping. Before any map is computed, the loaded model must
reproduce test macro-F1 0.5756 on the ISIC 2019 test split. If it does not,
stop.

**Attention maps, the details §5 leaves open:**
- Maps are compared with ground truth at model-input resolution: CV-4 at
  224×224, CV-3 at 512×512, CV-2 at the original image size. The CAM is
  upsampled bilinearly and ground-truth masks nearest-neighbour.
- Pointing game: the location of the map's maximum. An all-zero map counts
  as a miss.
- Energy fraction: the map's sum inside the ground truth divided by its total
  sum. An all-zero map scores 0.
- IoU@0.5: the map normalised to [0, 1] by its maximum, thresholded at 0.5,
  as `gradcam_mask_iou` does.
- CV-2 maps are standard Grad-CAM over the whole layer2 map, so energy is
  measured against the whole image.
- CV-3: the predicted mask is the region of probability ≥ 0.5. If it is empty
  the map is undefined; the count of such images is reported, and they are
  excluded.
- Sanity check: 50 items sampled with seed 42 (images for CV-3 and CV-4,
  true-positive boxes for CV-2). The target layer is re-initialised with
  `reset_parameters()` under seed 42. The target (class, mask region or box)
  is held at the original model's. Spearman correlation is computed on the
  native CAM grid. An undefined correlation (a constant map) counts as 0.

---

## 4. Performance decision rule (committed before running)

| Task | "Matches" if | Metric source |
|---|---|---|
| CV-2 | image-level recall ≥ **0.78** (0.81 − 0.03) **and** burden median ≤ 1, p90 ≤ 2 | `scripts/measure_cv2_revised_metrics.py` at conf 0.25. The Faster R-CNN predictions must be exported at conf 0.001 in the same `predictions.csv` schema as B1/E. |
| CV-3 | test Dice ≥ **0.854** (0.864 − 0.01) | `scripts/evaluate_cv3.py`, threshold 0.5 |
| CV-4 | matches by construction | — |

| Outcome | Condition | Action |
|---|---|---|
| **A** | CV-2 and CV-3 both match | Claim: one backbone serves all three tasks at production-level performance. |
| **B** | CV-3 matches, CV-2 does not | Finding: a general-purpose two-stage detector on ResNet-50 does not match a detector designed for small objects (YOLO11s) at iToBoS lesion scale (median area 0.00097). The production pipeline keeps YOLO for CV-2. |
| **C** | CV-3 does not match | Finding: the pretrained encoder does not beat a from-scratch U-Net at this resolution. Report it. |

CV-2 recall ≥ 0.90 (the original Section 22 gate) would be reported as a
notable result, but it does not change the outcome letter.

---

## 5. Attention maps

### 5.1 What "attention map" means here

ResNet-50 is a convolutional network and has **no attention layers**, so no
attention weights exist to read out. The maps here are **Grad-CAM saliency
maps**: how strongly each spatial region pushes the chosen output, computed
from gradients at a chosen feature layer. They are called "attention maps" in
the write-up, with this definition stated once. True attention maps would
need a transformer backbone (ViT or Swin). That would change the backbone and
is out of scope here.

### 5.2 Method per task

| Task | Method | Target output | Layer | Why this layer |
|---|---|---|---|---|
| CV-4 | Grad-CAM (Selvaraju et al., 2017) | predicted-class logit | `layer4` (7×7 at 224) | Standard. Reuses `src/explainability/gradcam.py`, whose head is currently hardcoded to PAD-UFES and must take the ISIC head as a parameter. |
| CV-3 | Seg-Grad-CAM (Vinogradova et al., 2020) | sum of mask logits inside the predicted mask | encoder `layer4` | Same layer as CV-4, so the two maps can be compared. |
| CV-2 | Grad-CAM for each detected box | that box's lesion score (through RoI-Align) | backbone `layer2` (stride 8) | At stride 32 (`layer4`), a median lesion (~30 px at 1280) covers about one cell and the map would be uninformative. |

Each map is fixed to one method and one layer. No layer or method sweep.

### 5.3 Measuring the maps against ground truth

| Task | Evaluation set | Ground truth | Metrics |
|---|---|---|---|
| CV-4 | HAM10000 images in the ISIC 2019 **test** split (1,451) | Tschandl masks (`data/raw/ham10000_masks/`, load with `.convert("L")`) | Pointing game (does the map's peak fall inside the mask?); energy-in-mask fraction; IoU at threshold 0.5 (existing `gradcam_mask_iou`) |
| CV-3 | ISIC 2018 Task 1 test (260) | GT masks | Same three metrics |
| CV-2 | True-positive boxes on iToBoS val | GT boxes | Pointing game (peak inside the matched GT box); energy-in-box fraction |

**Sanity check (required before any claim), from Adebayo et al., 2018:** on a
fixed 50-image sample per task, re-initialise the target layer's weights
randomly and recompute the maps. If the Spearman rank correlation between
original and randomised maps has median ≥ 0.5, the maps are not tied to what
the model learned. Report that, and make no lesion-focus claims for that task.

**Claim rule:** "Model X attends to the lesion" may be written only if
X passes the sanity check **and** pointing game ≥ 0.80 **and**
energy-in-lesion ≥ 0.50. Otherwise, report the numbers as they are, as a
limitation.

**Figures:** one contact sheet per task, 20 images chosen by fixed seed
(`random_state=42`). For CV-4, 10 correct and 10 misclassified. For CV-2, 10
true positives and 10 false positives. No hand-picking.

---

## 6. Anti-rabbit-hole boundary

After the two training runs and the map evaluation, apply Sections 4 and 5.3
and stop. Do **not**:
- tune anchors, try tiling/SAHI, or swap in RetinaNet/FCOS if CV-2 misses
  (tiling is already the deferred CV-2 lever, with its own re-entry gate);
- sweep CAM layers or switch to Grad-CAM++ / Score-CAM / other methods to get
  better-looking maps;
- add seeds, change the evaluation sets, or loosen the claim thresholds;
- investigate single images where a map looks odd.

---

## 7. Deliverables

- `evaluation/academic_r50/performance.csv`: one row per task, Section 4
  metrics next to the baseline.
- `evaluation/academic_r50/attention_metrics.csv`: per task, the Section 5.3
  metrics and the sanity-check correlation.
- `evaluation/academic_r50/attention_{cv2,cv3,cv4}.jpg`: contact sheets.
- `docs/academic_resnet50_three_task_result.md`: outcome letter, claims
  allowed or withheld, limitations.
