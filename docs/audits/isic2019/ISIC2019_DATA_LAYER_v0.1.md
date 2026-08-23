# DermaSense — ISIC 2019 Data Layer v0.1

**Status:** FROZEN  
**Version:** v0.1  
**Dataset:** ISIC 2019 Training Dataset  
**Image Domain:** Dermoscopic  
**Primary Identity Unit:** `lesion_id`  
**Patient-Level Guarantee:** Not available from supplied metadata

---

## 1. Purpose

This document freezes the validated ISIC 2019 data layer used by
DermaSense.

It records:

- the validated source dataset,
- native labels,
- image and lesion counts,
- identity limitations,
- archive-to-metadata correspondence,
- lesion-level split policy,
- train/validation/test artifacts,
- evaluation eligibility,
- and known limitations.

This document is a data-layer contract.

It does **not** define:

- model architecture,
- preprocessing strategy,
- augmentation,
- classification loss,
- risk thresholds,
- severity rules,
- clinical diagnosis,
- or the final DermaSense risk taxonomy.

---

# 2. Source Dataset

## 2.1 Dataset

ISIC 2019 Training Dataset.

The dataset contains dermoscopic skin-lesion images with native diagnostic
annotations and accompanying metadata.

---

## 2.2 Validated Source Size

| Property | Value |
|---|---:|
| Total images | **25,331** |
| Unique image IDs | **25,331** |
| Identified images | **23,247** |
| Unknown-lesion-ID images | **2,084** |
| Identified source lesions | **11,847** |

The raw dataset was independently validated before construction of the
manifest and splits.

---

# 3. Native Diagnostic Classes

ISIC 2019 provides the following native diagnostic classes:

| Native class | Image count |
|---|---:|
| MEL | 4,522 |
| NV | 12,875 |
| BCC | 3,323 |
| AK | 867 |
| BKL | 2,624 |
| DF | 239 |
| VASC | 253 |
| SCC | 628 |
| UNK | 0 |
| **Total** | **25,331** |

Every image has exactly one native diagnostic label.

The raw-data audit verified:

- zero images with no native label,
- zero images with multiple native labels,
- zero `UNK`-positive images.

---

# 4. Image Identity

## 4.1 Image Identity

The `image` identifier is unique across the complete dataset.

Validation confirmed:

```text
Ground-truth image IDs: 25,331
Metadata image IDs:     25,331
Archive image IDs:      25,331