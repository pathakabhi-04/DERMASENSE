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

## 2. Design decision: what is merged and why

In the product pipeline, CV-2 (detection, YOLO11s on iToBoS), CV-3
(segmentation, U-Net on ISIC 2018 Task 1) and CV-4 (classification, ResNet-50
on ISIC 2019) are three separate models. The academic track asked whether one
model could do all three. **Decision: merge segmentation and classification
into one shared-encoder model. Keep learned detection out.**

### 2.1 Why not all three in one model

1. **No dataset has all three labels on the same images.** Each of our tasks
   was trained on a different dataset carrying one label type: iToBoS has
   boxes only (no diagnosis, no masks), ISIC 2018 Task 1 has masks only, and
   ISIC 2019 has diagnoses only. A single forward pass that learns boxes, masks
   and diagnoses together needs all three on one image. The obvious in-house
   join (ISIC 2018 masks × ISIC 2019 labels) was checked and fails (Section 3).
   No public dataset we know of has wide-field images with a box *and* a
   diagnosis per lesion.

2. **Detection and diagnosis work at incompatible scales.** iToBoS lesions
   have a median area of 0.00097 of the image (a few tens of pixels across in
   a wide-field photo). That scale caps CV-2 image-level recall at 0.81
   (`docs/cv2_status.md`). A lesion that small carries essentially no
   diagnostic texture, so classification cannot run on the detection pass
   anyway. Even a "single" model would run twice (full image → detect → crop
   → classify), so merging detection saves weights, not passes.

3. **Training on the three datasets separately (partial labels) confounds
   the result.** One backbone could be trained by alternating iToBoS,
   ISIC 2018 and ISIC 2019 batches, each supervising only its own head. But the
   three datasets differ in domain as well as task (wide-field clinical TBP vs
   dermoscopic close-ups). Any gain or loss could then come either from
   sharing across tasks or from the domain shift, and the experiment could not
   separate the two. Negative transfer from the tiny-lesion detection task is
   also the most likely outcome. That makes this a second-stage experiment,
   not the first.

4. **A learned box head on lesion-centric images is trivial.** On datasets
   that do have masks and diagnoses (HAM10000, ISIC 2017, PH2), every image
   has one centred lesion. A box head would learn "the lesion is in the
   middle" and give no measurable result.

### 2.2 Why segmentation + classification is the merge we made

1. **Same image, same scale.** In the pipeline, CV-3 and CV-4 already run in
   sequence on the same lesion-centric crop, so a shared encoder changes no
   interface. Only CV-2 works on a different image, the wide-field photo.
2. **Both labels exist on the same images, without leakage.** HAM10000 plus
   the Tschandl masks gives 10,015 images with a mask and an 8-class diagnosis
   each, and all of them already sit in our lesion-grouped ISIC 2019 split
   (Section 3).
3. **Only one thing changes.** Holding the dataset, backbone and split fixed,
   the only difference between the arms is whether the tasks share an encoder.
   Any effect can be attributed to that, which the full three-task merge
   cannot offer (2.1.3).
4. **It tests the implicit version of a hypothesis we already falsified
   explicitly.** Commit `9c1b0dc` showed that hand-crafted ABCD geometry
   measured from U-Net masks does not help diagnosis (asymmetry AUC 0.44,
   border-solidity 0.48, both in the wrong direction). Joint training asks
   instead whether *learning* to segment shapes the encoder's features
   usefully for diagnosis, without relying on measurements from a predicted
   border. It also uses dermatologist-curated masks rather than U-Net outputs.
5. **Localisation still comes out.** The box is taken as the tight box around
   the predicted mask, so the model reports box, mask and diagnosis. Only the
   box is not a separately learned head.

**Re-entry for detection:** the iToBoS partial-label version (2.1.3) gets its
own spec only if this experiment ends in outcome A or B (Section 8).

---

## 3. Why this dataset (data check, done before writing this spec)

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

## 4. Scope

**In:** segmentation + classification from one encoder. A bounding box is
reported as a derived output (tight box of the predicted mask), not a learned
head.

**Out, and why:**
- **Learned detection head, and iToBoS.** See Section 2.1.
- **ISIC 2017 / PH2 / Dermofit.** Extra external sets need de-duplication
  against HAM and widen the experiment. Not in this spec.
- **Loss-weight tuning, GradNorm / uncertainty weighting, backbone changes.**

---

## 5. Arms

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

### 5.1 Training settings (fixed 2026-09-26, before the first run)

Section 5 says the arms share these settings but does not give values. They
are fixed here, before any run, and are identical for S, C and J. The CV-3 and
CV-4 baselines disagree on some of them, so each choice below says which one
it follows.

| Setting | Value | Source |
|---|---|---|
| Encoder | torchvision ResNet-50, `ResNet50_Weights.DEFAULT` | CV-4 |
| Decoder (S, J) | U-Net: 5 blocks (256, 128, 64, 32, 16 ch), nearest ×2 upsample, skips from stem / layer1–3, 2× conv3×3-BN-ReLU per block; 1×1 conv to 1 logit | — |
| Classifier (C, J) | GAP on layer4 → Linear(2048, 8) | CV-4 |
| Input | 512×512 squash resize, ImageNet mean/std | spec §5 (resolution); normalisation because the encoder is ImageNet-pretrained |
| Augmentation | hflip 0.5, vflip 0.5, rotation ±15°, colour jitter (0.10, 0.10, 0.10, 0.02); geometric ops applied jointly to the mask (nearest) | CV-4 (CV-3 used none) |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4, no scheduler, no grad clipping | CV-4 (CV-3: same lr) |
| Batch size | 16 | between CV-4 (32 at 224) and CV-3 (8 at 512) |
| Epochs | 30 for all arms; best-val checkpoint | — (CV-4: 10 on ~18k images; CV-3: 50 on ~2k images) |
| Precision | fp16 autocast + GradScaler on CUDA | CV-3 |
| Class weights (C, J) | sqrt inverse frequency, mean-normalised, from HAM train | CV-4 |
| Seg metric | thresholded Dice at 0.5, smooth 1, per image, at 512×512; HAM masks resized nearest | CV-3 |
| Pairing | For a given seed, all arms get the same shuffle order and augmentations (dedicated DataLoader generator), and J starts from the same classifier-head init as C and the same decoder init as S | — |

**PAD-UFES transfer (Section 6, secondary).** This is the C1 protocol with
C1's hyperparameters unchanged: layer4 + a new 6-class head, lr 1e-5 / 1e-4,
weight decay 1e-4, batch 32, ≤ 30 epochs, patience 7, input 224×224, best by
val macro-F1, scored once on PAD test. It uses the run's own seed so that C
and J transfers are paired. The encoders are trained at 512 but fine-tuned at
C1's 224; this mismatch is the same for C and J. PAD images are stored in the
bundle already resized to 224×224. Train-time augmentation therefore runs
after the resize, where C1 ran it before; this is also the same for C and J.

**ISIC 2018 external test.** The images are stored already resized to
512×512, using the resize CV-3 evaluation applies (cv2 linear / nearest), so
the Dice matches `scripts/evaluate_cv3.py` on main.

Code: `src/academic/`, `scripts/academic_joint/`.

---

## 6. Metrics

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

## 7. Preconditions (stop if any fails)

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

## 8. Decision rule (committed before running)

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

## 9. Anti-rabbit-hole boundary

After the 9 runs, apply Section 8 and stop. Do **not**:
- sweep λ, loss-weighting schemes, decoder variants, or resolutions;
- add seeds, datasets, or proxy metrics to move a borderline result;
- open per-class or per-image investigations into why J won or lost.

A borderline or surprising result is documented as-is, with its limitation,
and the track moves on.

---

## 10. Deliverables

- `evaluation/academic_joint/results.csv` — one row per (arm, seed) with all
  Section 6 metrics.
- `docs/academic_joint_seg_cls_result.md` — outcome letter, paired table,
  limitations.
