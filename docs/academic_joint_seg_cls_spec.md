# Academic Track — Joint Segmentation + Classification (Single Model)

**Status:** SPEC — preconditions passed (2026-09-25); runs not started.
**Purpose:** Test whether one shared-encoder model can replace the separate
CV-3 (segmentation) and CV-4 (classification) models, on images that carry
both labels.

---

## 1. Question

On HAM10000, does a single ResNet-50-encoder model with a segmentation head
and a classification head **match** separate single-task models trained on the
same images, and does the segmentation objective **improve** classification
(in-domain and on PAD-UFES smartphone images)?

One question, one dataset, three arms, fixed seeds. Nothing else.

---

## 2. Why this dataset (data check, done before writing this spec)

No dataset we use has boxes + masks + diagnosis on the same images. The two
obvious in-house candidates were checked:

| Candidate | Result | Usable? |
|---|---|---|
| ISIC 2018 Task 1 masks joined to ISIC 2019 labels | 772 / 2,594 images have a label; only NV (608) and MEL (164) | No — 2 classes, 30% coverage |
| HAM10000 inside our ISIC 2019 splits | All 10,015 present; 7,470 lesions; lesion-grouped (0 lesions cross splits) | **Yes**, once masks are added |

HAM10000 lesion masks (Tschandl et al., Harvard Dataverse,
`HAM10000_segmentations_lesion_tschandl`) supply the missing label. Using our
existing ISIC 2019 split restricted to HAM keeps it leakage-free against every
existing checkpoint.

**HAM split (inherited, seed 42):**

| Class | Train | Val | Test |
|---|---|---|---|
| AK | 92 | 32 | 6 |
| BCC | 377 | 70 | 67 |
| BKL | 777 | 160 | 162 |
| DF | 78 | 20 | 17 |
| MEL | 766 | 195 | 152 |
| NV | 4,695 | 1,007 | 1,003 |
| SCC | 146 | 25 | 26 |
| VASC | 100 | 24 | 18 |
| **Total** | **7,031** | **1,533** | **1,451** |

AK (6) and SCC (26) test counts are small; per-class F1 for them is reported
but is not decision-relevant. Macro-F1 is the decision metric regardless.

**External segmentation test:** the CV-3 ISIC 2018 Task 1 test split (260
images). Verified 0 of the 2,594 ISIC 2018 Task 1 images carry a HAM lesion ID,
so it is disjoint from HAM.

---

## 3. Scope

**In:** segmentation + classification from one encoder. A bounding box is
reported as a derived output (tight box of the predicted mask), not a learned
head.

**Out, and why:**
- **Learned detection head.** Every HAM image has one centred lesion; a box
  head would learn trivial localisation and add no measurable result.
  Wide-field detection (iToBoS) is a separate, later spec — only if this one
  ends in outcome A or B (Section 7).
- **ISIC 2017 / PH2 / Dermofit.** Extra external sets need de-duplication
  against HAM and widen the experiment. Not in this spec.
- **Loss-weight tuning, GradNorm / uncertainty weighting, backbone changes.**

---

## 4. Arms

All arms: ResNet-50 ImageNet-pretrained encoder (same as the CV-4 baseline),
input 512×512 (CV-3 resolution), same augmentation, same optimizer, same epoch
budget, same checkpoint-selection rule (best val on that arm's own primary
metric; J uses val macro-F1).

| Arm | Heads | Loss |
|---|---|---|
| **S** — seg only | U-Net decoder on ResNet-50 encoder | 0.5 BCE + 0.5 Dice (CV-3 loss) |
| **C** — cls only | GAP → linear, 8 classes | Weighted CE (CV-4 loss) |
| **J** — joint | Both, shared encoder | L_cls + 1.0 · L_seg — λ fixed at 1.0, **not swept** |

**Seeds:** 42, 43, 44 per arm → 9 runs total. Fixed; not enlarged.

---

## 5. Metrics

| Metric | Split | Role |
|---|---|---|
| Macro-F1 (8-class) | HAM test | **Primary — classification** |
| Mean Dice | HAM test (Tschandl masks) | **Primary — segmentation** |
| Mean Dice | ISIC 2018 Task 1 test (260) | Secondary — external seg |
| Macro-F1 | PAD-UFES test, after the CV-4 C1 partial fine-tune protocol (same script, same hyperparameters) applied to the C and J encoders | Secondary — smartphone transfer |

Existing baselines (CV-4 test macro-F1 0.5768 on full ISIC 2019; CV-3 Dice
0.8640 on ISIC 2018) are **context only** — different training data, not
comparable. The comparisons that decide the outcome are J vs S and J vs C
within this spec.

All comparisons are paired by seed (J_seed − C_seed, J_seed − S_seed), reported
as mean and per-seed values.

---

## 6. Preconditions (stop if any fails)

1. Masks downloaded; licence permits academic use; recorded in `docs/`.
2. Mask count matches: ≥ 99% of the 10,015 HAM image IDs have a mask. If
   not, stop and report — do not substitute or impute.
3. Spot-check 20 random masks overlaid on images (contact sheet in
   `evaluation/academic_joint/`). If > 2 are clearly wrong, stop and report.

**Checked 2026-09-25 — all three pass.**

| Check | Result |
|---|---|
| Source | Harvard Dataverse `doi:10.7910/DVN/DBW86T` (v4.0), file `HAM10000_segmentations_lesion_tschandl.zip`, 10,808,743 bytes, md5 `6e8d252e09cfdb0189199f15985a5b84` |
| Local path | `data/raw/ham10000_masks/HAM10000_segmentations_lesion_tschandl/{image_id}_segmentation.png` (original zip kept alongside as `masks.zip`) |
| Licence | CC BY-NC 4.0 — same terms as the HAM10000 images we already use. Academic use is fine; commercial use is not. |
| Citation | Tschandl et al., "Human-computer collaboration for skin cancer recognition", *Nature Medicine* 2020, doi:10.1038/s41591-020-0942-0 |
| Provenance | "Binary lesion-segmentations curated by a single dermatologist" (single annotator: no inter-rater agreement available) |
| Coverage | 10,015 masks = 10,015 HAM IDs in our splits; 0 missing, 0 extra, 0 duplicates |
| Format | 450×600, binary; 0 empty masks; 2 full-frame masks (ISIC_0029819, ISIC_0026042), visually confirmed as lesions filling the frame |
| Spot-check | 20 masks (2 per class + 4 random): 0 wrong |

**Loader requirement:** 402 of the PNGs are palette mode (`P`) with an
inverted palette (index 0 = white). `np.array(Image.open(p))` reads those
masks inverted, without any error. Always load with
`Image.open(p).convert("L")`, then threshold at > 127.

---

## 7. Decision rule (committed before running)

Margins: classification −0.02 macro-F1, segmentation −0.01 Dice.

- **J matches** ⇔ mean(J − C) macro-F1 ≥ −0.02 **and** mean(J − S) Dice ≥ −0.01.
- **J improves classification** ⇔ J − C > 0 in all 3 seeds **and** mean ≥ +0.02.

| Outcome | Condition | Action |
|---|---|---|
| **A** | Matches and improves | Headline result: segmentation supervision helps diagnosis. Write up. Unlocks the iToBoS partial-label spec. |
| **B** | Matches, no improvement | Result: one model does both at no accuracy cost (half the parameters, one forward pass). Write up. Unlocks the iToBoS spec. |
| **C** | Fails to match | Result: negative transfer at λ = 1. Write up as a finding. **Stop** — no λ sweep, no rescue attempts. |

PAD-UFES transfer uses the same improvement rule and is reported alongside,
but it does not change the outcome letter.

---

## 8. Anti-rabbit-hole boundary

After the 9 runs, apply Section 7 and stop. Do **not**:
- sweep λ, loss-weighting schemes, decoder variants, or resolutions;
- add seeds, datasets, or proxy metrics to move a borderline result;
- open per-class or per-image investigations into why J won or lost.

A borderline or surprising result is documented as-is, with its limitation,
and the track moves on.

---

## 9. Deliverables

- `evaluation/academic_joint/results.csv` — one row per (arm, seed) with all
  Section 5 metrics.
- `docs/academic_joint_seg_cls_result.md` — outcome letter, paired table,
  limitations.
