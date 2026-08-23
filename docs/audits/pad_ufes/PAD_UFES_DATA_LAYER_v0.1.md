# DermaSense — PAD-UFES-20 Data Layer v0.1

**Status:** Frozen  
**Dataset:** PAD-UFES-20  
**Domain:** Smartphone / clinical skin images  
**Purpose:** Validated CV data layer for DermaSense

---

## 1. Scope

This document freezes the validated PAD-UFES-20 data layer used by DermaSense.

The scope includes:

- raw dataset integrity;
- image/metadata correspondence;
- image, lesion, and patient identity;
- native diagnostic labels;
- label-strength metadata;
- operational lesion identity;
- patient-level train/validation/test partitioning;
- independent split validation.

This document does **not** define the final DermaSense risk taxonomy or training target.

Those decisions belong to the CV Dataset Specification v1.0.

---

## 2. Raw Dataset Validation

The downloaded PAD-UFES-20 dataset was validated before manifest construction.

### Validated counts

| Quantity | Count |
|---|---:|
| Images | 2,298 |
| Unique image IDs | 2,298 |
| Unique source lesion IDs | 1,641 |
| Unique patients | 1,373 |
| Duplicate image IDs | 0 |
| Missing images | 0 |
| Extra images | 0 |

The raw dataset passed image/metadata correspondence validation.

**Raw validation status: PASS**

---

## 3. Native Diagnostic Distribution

The dataset contains six native diagnostic labels.

| Native diagnosis | Images |
|---|---:|
| ACK | 730 |
| BCC | 845 |
| MEL | 52 |
| NEV | 244 |
| SCC | 192 |
| SEK | 235 |
| **Total** | **2,298** |

These labels are preserved in the manifest.

No DermaSense risk-category label is substituted at the data-layer stage.

---

## 4. Label Strength

The dataset contains two observed label-strength categories.

| Label strength | Images |
|---|---:|
| biopsy_backed | 1,342 |
| clinical | 956 |
| **Total** | **2,298** |

The distinction is retained because native diagnostic labels do not have uniform evidentiary strength across the dataset.

The final CV training specification must explicitly determine how this information is used.

---

## 5. Image Domain

All PAD-UFES-20 images used in this data layer are categorized as:

```text
smartphone_clinical