# DermaSense — CV Dataset Specification v1.0

**Status:** Draft for freeze  
**Purpose:** Fixed contract for Stage 1 computer-vision development  
**Scope:** Stage 1 CV data only

---

## 1. Purpose

This document defines the computer-vision data contract for DermaSense Stage 1.

It converts the completed PAD-UFES-20 and ISIC 2019 dataset audits into an implementation-level specification.

Once frozen, downstream training code must consume the artifacts defined here rather than independently reconstructing dataset membership, lesion identity, or train/validation/test partitions.

This specification covers:

- dataset membership;
- dataset roles;
- native diagnostic labels;
- split artifacts;
- identity guarantees;
- evaluation eligibility;
- domain information;
- label-strength information;
- Stage 1 training targets;
- Stage 1 evaluation targets;
- leakage constraints.

This specification does **not** freeze:

- model architecture;
- image resolution;
- augmentation policy;
- optimizer;
- loss weighting;
- calibration method;
- uncertainty method;
- deployment thresholds;
- final LOW/MEDIUM/HIGH risk thresholds.

---

## 2. Dataset Sources

Stage 1 uses exactly two validated datasets.

| Dataset | Domain | Images | Role |
|---|---|---:|---|
| PAD-UFES-20 | Smartphone / clinical | 2,298 | Primary smartphone-domain source |
| ISIC 2019 | Dermoscopic | 25,331 | Primary dermoscopic source |

No other dataset is part of the v1.0 training/evaluation contract.

In particular:

- HAM10000 is not added separately to ISIC 2019;
- ISIC 2019 is treated as the validated dataset artifact already audited;
- unvalidated external datasets are not permitted in the v1.0 evaluation set.

---

## 3. Frozen Dataset Artifacts

### 3.1 PAD-UFES-20

Manifest:

```text
data/manifests/pad_ufes_manifest.csv
```

Splits:

```text
data/splits/pad_ufes/train.csv
data/splits/pad_ufes/val.csv
data/splits/pad_ufes/test.csv
data/splits/pad_ufes/split_summary.csv
```

Audit:

```text
docs/audits/pad_ufes/PAD_UFES_DATA_LAYER_v0.1.md
```

### 3.2 ISIC 2019

Manifest:

```text
data/manifests/isic2019_manifest.csv
```

Splits:

```text
data/splits/isic2019/train.csv
data/splits/isic2019/val.csv
data/splits/isic2019/test.csv
data/splits/isic2019/split_summary.csv
```

Audits:

```text
docs/audits/isic2019/ISIC2019_DATA_LAYER_v0.1.md
docs/audits/isic2019/ISIC2019_IDENTITY_AUDIT_v0.1.md
docs/audits/isic2019/ISIC2019_SPLIT_POLICY_v0.1.md
```

---

## 4. PAD-UFES-20 Contract

### 4.1 Dataset identity

The validated PAD-UFES-20 data layer contains:

| Quantity | Count |
|---|---:|
| Images | 2,298 |
| Patients | 1,373 |
| Source lesion IDs | 1,641 |
| Operational lesions | 1,891 |

The raw `lesion_id` field is not globally unique across patients.

Therefore the source lesion ID must not be used as the sole global split key.

The operational lesion identity is the frozen identity used for leakage prevention.

### 4.2 Native diagnoses

PAD-UFES-20 native diagnoses are:

```text
ACK
BCC
MEL
NEV
SCC
SEK
```

Native image counts:

| Diagnosis | Images |
|---|---:|
| ACK | 730 |
| BCC | 845 |
| MEL | 52 |
| NEV | 244 |
| SCC | 192 |
| SEK | 235 |
| **Total** | **2,298** |

Native labels are preserved.

They are not silently converted into DermaSense risk categories at the dataset layer.

### 4.3 Label strength

PAD-UFES-20 contains:

```text
biopsy_backed
clinical
```

Counts:

| Label strength | Images |
|---|---:|
| biopsy_backed | 1,342 |
| clinical | 956 |
| **Total** | **2,298** |

The `label_strength` field remains available to downstream CV code.

No weighting or filtering policy is frozen by this specification.

### 4.4 Image domain

All PAD-UFES-20 images are classified as:

```text
smartphone_clinical
```

This domain metadata remains available for domain-specific analysis.

### 4.5 Frozen split

PAD-UFES-20 uses a patient-level split.

| Split | Patients | Images | Operational lesions |
|---|---:|---:|---:|
| Train | 961 | 1,610 | 1,323 |
| Val | 204 | 336 | 278 |
| Test | 208 | 352 | 290 |
| **Total** | **1,373** | **2,298** | **1,891** |

The split guarantees:

```text
Train ∩ Val patients  = 0
Train ∩ Test patients = 0
Val ∩ Test patients   = 0

Train ∩ Val lesions   = 0
Train ∩ Test lesions  = 0
Val ∩ Test lesions    = 0

Train ∩ Val images    = 0
Train ∩ Test images   = 0
Val ∩ Test images     = 0
```

Every PAD-UFES-20 image occurs exactly once across the three splits.

### 4.6 Evaluation eligibility

PAD-UFES-20 validation and test images are evaluation eligible.

The frozen split artifacts and independent split validator are authoritative.

### 4.7 Operational lesion identity

The source `lesion_id` field was found to be reused across multiple patients.

The audit identified 250 reused source lesion IDs.

Therefore the split system uses an operational lesion identity based on the patient + source lesion relationship.

Conceptually:

```text
operational_lesion_uid = patient_id + source_lesion_id
```

This produces:

```text
1,891 operational lesions
```

The operational lesion is the unit used for lesion-level leakage checks.

Downstream split generation must not revert to raw `lesion_id` as the sole grouping key.

---

## 5. ISIC 2019 Contract

### 5.1 Dataset identity

The validated ISIC 2019 data layer contains:

| Quantity | Count |
|---|---:|
| Images | 25,331 |
| Identified images | 23,247 |
| Unknown-lesion-ID images | 2,084 |
| Identified lesion IDs | 11,847 |

All 25,331 physical JPEG files passed raw image validation.

All images passed image decoding validation.

### 5.2 Native diagnoses

ISIC 2019 native diagnoses are:

```text
AK
BCC
BKL
DF
MEL
NV
SCC
VASC
```

Native image counts:

| Diagnosis | Images |
|---|---:|
| AK | 867 |
| BCC | 3,323 |
| BKL | 2,624 |
| DF | 239 |
| MEL | 4,522 |
| NV | 12,875 |
| SCC | 628 |
| VASC | 253 |
| **Total** | **25,331** |

Every image has exactly one native diagnostic label.

`UNK` contains zero positive samples in the validated ground truth.

Native labels are preserved.

### 5.3 Lesion identity

ISIC 2019 contains:

- 11,847 identified lesion IDs;
- 6,788 single-image lesions;
- 5,059 multi-image lesions.

The audit found zero identified lesions with multiple native diagnoses.

Therefore every identified lesion has a consistent native diagnosis.

### 5.4 Unknown lesion IDs

There are:

```text
2,084
```

images without an identified lesion ID.

These represent:

```text
8.23%
```

of the complete ISIC 2019 image set.

The unknown-ID images cannot be proven lesion-disjoint against other datasets or split members using the supplied metadata.

Therefore they have a restricted role.

### 5.5 Frozen split

ISIC 2019 uses lesion-level grouping for identified lesions.

| Split | Images | Identified lesions | Unknown-ID images |
|---|---:|---:|---:|
| Train | 18,402 | 8,291 | 2,084 |
| Val | 3,375 | 1,776 | 0 |
| Test | 3,554 | 1,780 | 0 |
| **Total** | **25,331** | **11,847** | **2,084** |

The split guarantees:

```text
Train ∩ Val images = 0
Train ∩ Test images = 0
Val ∩ Test images   = 0

Train ∩ Val identified lesions = 0
Train ∩ Test identified lesions = 0
Val ∩ Test identified lesions   = 0
```

All 11,847 identified lesions are partitioned exactly once.

All 2,084 unknown-ID images are assigned to training.

No unknown-ID image occurs in validation or test.

### 5.6 Evaluation eligibility

ISIC 2019 evaluation eligibility is restricted to identified-lesion images.

Therefore:

```text
Validation eligibility = identified lesion ID required
Test eligibility       = identified lesion ID required
```

The 2,084 unknown-ID training images are not evaluation eligible.

### 5.7 Patient-level limitation

The supplied ISIC 2019 metadata contains no `patient_id` field.

Therefore patient-level independence cannot be mechanically verified.

This limitation must remain explicit in all reports.

The ISIC 2019 split provides lesion-level independence, not a verified patient-level guarantee.

---

## 6. Cross-Dataset Domain Contract

The two datasets represent different image domains.

```text
PAD-UFES-20 → smartphone clinical imagery
ISIC 2019    → dermoscopic imagery
```

Domain information must not be discarded during dataset loading.

At minimum, downstream datasets must preserve:

```text
dataset_id
image_domain
native_diagnosis
```

where available.

Cross-domain performance must be reported separately when domain-specific evaluation is performed.

A single pooled score must not be interpreted as proof of equal performance across smartphone and dermoscopic imagery.

---

## 7. Stage 1 Training Target

Stage 1 CV development operates on the **native diagnostic labels** supplied by the validated datasets.

The dataset layer therefore exposes:

```text
native_diagnosis
```

rather than a prematurely collapsed:

```text
LOW
MEDIUM
HIGH
```

risk label.

The exact classifier formulation, class mapping between datasets, loss design, and architecture are outside this dataset specification.

---

## 8. Native Diagnosis vs Risk Category

Native diagnosis and downstream clinical risk are separate concepts.

The intended conceptual flow is:

```text
Dataset evidence
      ↓
Native diagnostic prediction
      ↓
Model confidence / uncertainty
      ↓
Downstream risk reasoning
      ↓
LOW / MEDIUM / HIGH
```

The dataset layer must not pretend that native diagnostic classes are themselves final patient-risk categories.

This specification does not define a fixed mapping such as:

```text
ACK = LOW
BCC = HIGH
MEL = HIGH
```

or any equivalent mapping.

Any risk mapping must be defined and audited separately by the downstream risk layer.

---

## 9. Evaluation Policy

Validation and test sets are immutable evaluation artifacts.

Training code must not:

- add training images to validation or test;
- reconstruct alternative random splits;
- use evaluation images for training;
- use test labels for model selection;
- move unknown-ID ISIC images into evaluation;
- regroup PAD-UFES images using raw `lesion_id`;
- ignore the frozen operational lesion identity.

All reported Stage 1 benchmark results must identify, where applicable:

```text
dataset
split
domain
native diagnostic target
evaluation eligibility policy
```

---

## 10. Leakage Policy

The following are prohibited.

### 10.1 Image leakage

The same image ID must never occur in more than one split.

### 10.2 Lesion leakage

For PAD-UFES-20, operational lesions must never cross splits.

For ISIC 2019, identified lesion IDs must never cross splits.

### 10.3 Patient leakage

PAD-UFES-20 must retain its patient-level split guarantee.

For ISIC 2019, patient-level independence must not be claimed because patient identifiers are unavailable.

### 10.4 Evaluation leakage

Test images and test labels must not be used for:

- training;
- hyperparameter selection;
- model selection;
- threshold selection;
- architecture selection.

### 10.5 Metadata leakage

Metadata may be used only according to an explicitly documented downstream experiment.

The existence of metadata in the manifest does not automatically make every field a valid model input.

---

## 11. Required Dataset Runtime Assertions

Downstream CV loaders should verify the frozen contract before training.

At minimum:

```text
manifest exists
split exists
image ID exists
native diagnosis exists
dataset ID exists
image domain exists
```

For PAD-UFES-20:

```text
patient leakage = 0
operational lesion leakage = 0
image leakage = 0
```

For ISIC 2019:

```text
identified lesion leakage = 0
unknown-ID evaluation images = 0
image leakage = 0
```

A training or evaluation job should fail fast if these assertions are violated.

---

## 12. Evaluation Metrics

The dataset specification does not prescribe a single model architecture or optimizer.

However, Stage 1 evaluation must preserve class-wise performance visibility.

At minimum, reports should expose where applicable:

- macro F1;
- per-class precision;
- per-class recall;
- per-class F1;
- confusion matrix;
- ROC-AUC where mathematically appropriate;
- PR-AUC where appropriate;
- support per class.

Overall accuracy alone is insufficient for judging the Stage 1 diagnostic classifier.

Rare classes must not disappear behind aggregate accuracy.

---

## 13. Domain-Aware Evaluation

When both datasets are used in the same research pipeline, performance should be identifiable by domain.

At minimum:

```text
PAD-UFES-20 → smartphone / clinical
ISIC 2019    → dermoscopic
```

Cross-domain experiments must report the dataset-specific results separately before presenting pooled results.

A pooled result must not obscure domain-specific degradation.

---

## 14. What This Specification Does Not Freeze

This document intentionally does not freeze:

- CNN or transformer architecture;
- backbone;
- image resolution;
- augmentation;
- optimizer;
- learning-rate schedule;
- batch size;
- loss function;
- class weighting;
- oversampling;
- calibration algorithm;
- uncertainty algorithm;
- segmentation architecture;
- detection architecture;
- explainability method;
- final risk thresholds;
- clinical recommendation policy.

Those decisions belong to later engineering and modeling specifications.

---

## 15. Prohibited Dataset Modifications

Once v1.0 is frozen, downstream development must not silently modify:

- manifest membership;
- native labels;
- operational lesion identity;
- frozen split membership;
- evaluation eligibility;
- unknown-ID policy.

Any change requires:

1. a new specification version;
2. a new audit;
3. regenerated artifacts;
4. independent validation;
5. a new Git checkpoint.

---

## 16. Authoritative Artifacts

The following artifacts are authoritative for Stage 1 data membership and partitioning.

### PAD-UFES-20

```text
data/manifests/pad_ufes_manifest.csv
data/splits/pad_ufes/train.csv
data/splits/pad_ufes/val.csv
data/splits/pad_ufes/test.csv
data/splits/pad_ufes/split_summary.csv
```

### ISIC 2019

```text
data/manifests/isic2019_manifest.csv
data/splits/isic2019/train.csv
data/splits/isic2019/val.csv
data/splits/isic2019/test.csv
data/splits/isic2019/split_summary.csv
```

Scripts that independently recreate these artifacts must not replace the frozen CSV artifacts during training.

The committed artifacts are the source of truth.

---

## 17. Definition of Done

The Stage 1 dataset layer is considered frozen when:

- both datasets have passed raw-data validation;
- both manifests have been generated from validated sources;
- PAD-UFES-20 has passed independent patient/lesion/image split validation;
- ISIC 2019 has passed independent lesion/image split validation;
- ISIC 2019 unknown-ID policy is enforced;
- native diagnostic labels are preserved;
- evaluation eligibility is explicit;
- dataset domain is preserved;
- the CV dataset specification is committed;
- the repository checkpoint is pushed to the remote repository.

---

## 18. Current Status

The validated dataset artifacts currently satisfy the v1.0 data-layer requirements.

### PAD-UFES-20

```text
2,298 images
1,373 patients
1,891 operational lesions
patient-disjoint split
lesion-disjoint split
image-disjoint split
```

### ISIC 2019

```text
25,331 images
23,247 identified images
2,084 unknown-ID images
11,847 identified lesions
lesion-disjoint evaluation split
unknown-ID images restricted to training
patient-level guarantee unavailable
```

The dataset audits and independent split validators have passed.

**Specification:** `CV_DATASET_SPEC_v1.0`

**Status:** Ready for repository review and freeze.