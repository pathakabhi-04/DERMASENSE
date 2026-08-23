# DermaSense — ISIC 2019 Split Policy v0.1

## 1. Purpose

This document defines the fixed train/validation/test partition policy
for ISIC 2019 within the DermaSense computer-vision data layer.

The policy is derived from the completed ISIC 2019 Identity & Split Audit
and is intended to prevent image-level and lesion-level leakage.

This document defines splitting rules only. It does not define model
architecture, augmentation, loss functions, or clinical risk mapping.

---

## 2. Source Dataset

Dataset:

- ISIC 2019 Training Dataset

Validated source size:

- 25,331 images
- 25,331 unique image identifiers
- 11,847 identified source lesions
- 23,247 images with lesion IDs
- 2,084 images without lesion IDs

Native diagnostic classes:

- MEL
- NV
- BCC
- AK
- BKL
- DF
- VASC
- SCC

Image domain:

- Dermoscopic

---

## 3. Identity Hierarchy

ISIC 2019 does not provide a patient_id field in the supplied metadata.

Therefore patient-level split integrity cannot be directly guaranteed.

The strongest available identity key is:

    lesion_id

For images with a valid lesion_id, lesion identity is considered
the authoritative grouping unit for splitting.

---

## 4. Identified Lesion Records

There are:

- 23,247 images with identified lesion IDs
- 11,847 unique identified lesions

All identified lesions were found to have exactly one native diagnosis
in the completed identity audit.

A lesion may contain multiple images.

Therefore all images belonging to the same lesion must remain in the
same dataset split.

Required invariant:

    Train lesions ∩ Val lesions = ∅
    Train lesions ∩ Test lesions = ∅
    Val lesions ∩ Test lesions = ∅

No image-level random splitting is permitted for identified lesions.

---

## 5. Unknown Lesion-ID Records

There are 2,084 images without a source lesion_id.

These records cannot be proven lesion-disjoint from one another or from
other images using the supplied metadata.

Therefore:

- Unknown-lesion-ID images are NOT eligible for validation.
- Unknown-lesion-ID images are NOT eligible for test.
- They may be used as training-only data if explicitly marked as
  identity_unknown.

Unknown records must never silently enter validation or test sets.

---

## 6. Split Eligibility

### Identified records

Eligible for:

- Train
- Validation
- Test

Subject to lesion-level grouping.

### Unknown records

Eligible for:

- Train only

Not eligible for:

- Validation
- Test

---

## 7. Target Split Ratio

For identified lesions, the target partition is approximately:

- Train: 70%
- Validation: 15%
- Test: 15%

The split unit is the lesion, not the image.

Because lesions contain different numbers of images, exact image-level
70/15/15 proportions are not required.

The generator should prioritize:

1. zero lesion leakage
2. class coverage
3. approximate class balance
4. approximate image distribution

in that order.

---

## 8. Class Coverage

Every native diagnosis with sufficient identified-lesion support should
be represented in:

- Train
- Validation
- Test

The generator must report the resulting image, lesion, and patient
counts per native diagnosis.

No class should be silently removed to achieve a desired split ratio.

---

## 9. Unknown-ID Training Policy

Unknown-ID images may be included in training after the identified-lesion
split has been established.

They must retain explicit metadata:

    identity_status = unknown
    evaluation_eligible = false

This prevents them from being mistaken for verified independent lesions.

The use of unknown-ID images is therefore a training-data augmentation
decision, not an evaluation-data decision.

---

## 10. Evaluation Integrity

The locked validation and test sets must contain only images whose
lesion identity is known.

Evaluation must therefore be based on records for which lesion-level
disjointness can be mechanically verified.

The independent split validator must verify:

- image uniqueness
- complete partition of evaluation-eligible records
- no train/validation lesion overlap
- no train/test lesion overlap
- no validation/test lesion overlap
- no unknown-lesion records in validation
- no unknown-lesion records in test
- valid native diagnoses
- expected class coverage

---

## 11. Patient-Level Limitation

Because patient_id is absent from the supplied ISIC 2019 metadata:

    patient-level leakage cannot be ruled out.

This limitation must be documented in all downstream evaluation reports.

The ISIC split must therefore be described as:

    lesion-disjoint

and NOT:

    patient-disjoint

---

## 12. Non-Goals

This policy does not determine:

- classification taxonomy
- benign/suspicious/malignant mapping
- model architecture
- augmentation
- image preprocessing
- image quality thresholds
- detection/segmentation strategy
- uncertainty estimation
- temporal modeling
- clinical severity thresholds

Those decisions belong to their respective data/model layers.

---

## 13. Definition of Done

The ISIC split layer is considered valid only when:

1. The split generator completes successfully.
2. Train/validation/test contain no overlapping lesion IDs.
3. Validation and test contain zero unknown-lesion records.
4. All evaluation-eligible identified records are assigned exactly once.
5. Native class coverage is reported.
6. The independent validator passes.
7. The resulting artifacts are reproducible from the manifest and
   split-generation script.