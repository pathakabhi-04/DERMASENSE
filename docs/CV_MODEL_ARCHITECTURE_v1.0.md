# DermaSense — CV Model Architecture v1.0

**Status:** Architecture Freeze  
**Purpose:** Fixed architecture contract for Stage 1 native-diagnosis computer vision  
**Scope:** Stage 1 image classification architecture  
**Dataset Contract:** `docs/CV_DATASET_SPEC_v1.0.md`

---

## 1. Purpose

This document freezes the baseline computer-vision architecture for DermaSense Stage 1.

The purpose of this freeze is to separate:

1. architectural decisions;
2. dataset-contract decisions;
3. training-policy decisions;
4. experimental hyperparameters.

Once this architecture is frozen, GPU experiments must compare training configurations and model performance without silently changing the underlying architecture.

This document defines the architecture used for native diagnostic classification.

It does **not** define the downstream clinical risk engine.

---

# 2. Architectural Scope

Stage 1 classification operates on images supplied by the frozen CV dataset layer.

The high-level flow is:

```text
Validated Dataset
      ↓
Frozen Split
      ↓
CVDatasetTorch
      ↓
Image Transform
      ↓
Shared Visual Backbone
      ↓
512-D Feature Representation
      ↓
Dataset-Specific Native Diagnosis Head
      ↓
Native Diagnostic Logits