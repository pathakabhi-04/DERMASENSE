# Academic Track — One Backbone, Three Tasks (ResNet-50) + Attention Maps

**Status:** SPEC — not yet run.
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
