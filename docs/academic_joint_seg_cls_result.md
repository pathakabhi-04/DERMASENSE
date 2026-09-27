# Academic Track — Joint Segmentation + Classification: Result

**Spec:** `docs/academic_joint_seg_cls_spec.md` (settings in §5.1, fixed before the first run)
**Runs:** 9 (arms S, C, J × seeds 42, 43, 44), completed 2026-09-26 on RunPod
**Code:** `src/academic/`, `scripts/academic_joint/` at `f731d99`
**Data:** `academic_joint_bundle` (HAM 7,031 / 1,533 / 1,451; ISIC 2018 test 260; PAD-UFES 1,610 / 336 / 352)
**Raw numbers:** `evaluation/academic_joint/{results.csv, paired.csv, decision.json}`

---

## Outcome: **C — fails to match (negative transfer at λ = 1)**

The joint model misses **both** margins of the Section 8 rule:

| Criterion | Margin | Mean (J − paired baseline) | Result |
|---|---|---|---|
| Classification: J − C macro-F1, HAM test | ≥ −0.02 | **−0.0337** | fail |
| Segmentation: J − S Dice, HAM test | ≥ −0.01 | **−0.0126** | fail |
| Improvement: J − C > 0 in all seeds and mean ≥ +0.02 | — | 1 of 3 seeds > 0 | no |

As the spec requires, this is written up as a finding and the track **stops
here**: no λ sweep, no loss-weighting schemes, no rescue runs. The iToBoS
partial-label spec (unlocked only by outcome A or B) stays **locked**.

---

## 1. Per-arm results (mean ± SD over 3 seeds)

| Metric | S (seg only) | C (cls only) | J (joint) |
|---|---|---|---|
| HAM test macro-F1 (8-class) — **primary** | — | **0.6820** ± 0.0190 | 0.6483 ± 0.0282 |
| HAM test accuracy | — | 0.8488 ± 0.0028 | 0.8422 ± 0.0075 |
| HAM test Dice — **primary** | **0.9465** ± 0.0006 | — | 0.9339 ± 0.0020 |
| ISIC 2018 Task 1 test Dice (external) | **0.8865** ± 0.0079 | — | 0.8376 ± 0.0328 |
| PAD-UFES test macro-F1 after C1 transfer | — | 0.5709 ± 0.0057 | **0.5980** ± 0.0055 |
| Best epoch (of 30), per seed 42/43/44 | 29 / 26 / 25 | 21 / 16 / 9 | 20 / 20 / 26 |

## 2. Paired differences (decision inputs)

| Seed | J − C macro-F1 | J − S Dice (HAM) | J − S Dice (ISIC 2018) | J − C PAD macro-F1 |
|---|---|---|---|---|
| 42 | −0.0343 | −0.0134 | −0.0285 | +0.0233 |
| 43 | −0.0691 | −0.0141 | −0.0821 | +0.0191 |
| 44 | +0.0025 | −0.0103 | −0.0363 | +0.0391 |
| **Mean** | **−0.0337** | **−0.0126** | **−0.0490** | **+0.0271** |

## 3. Reading the result

- **Segmentation loss is consistent, not noise.** J is below S in every seed,
  on both test sets. On HAM, all three per-seed gaps are past the −0.01
  margin, and they are an order of magnitude larger than the seed-to-seed
  spread of S (SD 0.0006). The external ISIC 2018 gap is larger still
  (−0.049). So sharing the encoder costs the segmentation head accuracy, and
  the cost grows off-domain.
- **Classification loss is larger on average but less uniform.** Two of three
  seeds are past the −0.02 margin, and seed 43 is the largest (−0.069). Seed
  44 is level (+0.002). Per-class F1 averaged over seeds is lower for J in
  7 of 8 classes (NV is level). The largest drop is AK (0.348 → 0.205), but AK
  has 6 test images, so per-class values for it are not decision-relevant
  (spec §3).
- **The secondary PAD-UFES transfer goes the other way.** After the same C1
  fine-tune, J's encoder beats C's in all three seeds (mean +0.027, SD
  ≈ 0.006 in both arms). By the spec's improvement rule applied to PAD, this
  counts as an improvement. **Per §8 it does not change the outcome letter.**
  It is reported, not interpreted further (§9 forbids opening an
  investigation into why). Caveats: PAD test has 352 images, and MEL has only
  9 of them; the transfer fine-tunes at 224 px an encoder trained at 512 px.

- **Scope of the PAD-UFES result: seen-source adaptation, not
  generalisation** (note added 2026-09-28). The C1 protocol fine-tunes on
  PAD-UFES *train* before scoring PAD-UFES *test*, so PAD is a **seen** source
  when the metric is taken. Main's CV-4b experiments found exactly this split:
  adaptation to seen sources works (e.g. Fitzpatrick 0.93 AUC in the `pooled`
  fold), while transfer to an **unseen** capture source does not (0.63–0.67),
  across four experiments that isolated the gap as a data constraint
  (`docs/data_constraint_spec.md`, `cv4b_backbone_finetune_result.md` on
  main). J's +0.027 therefore says that a segmentation-shaped encoder adapts
  slightly better to a source it is then trained on. It says nothing about
  performance on an unseen capture source such as a user's phone, and must
  not be cited as evidence for that.

**One-line summary:** at λ = 1, one shared ResNet-50 encoder does both tasks
worse in-domain than two specialists (−0.034 macro-F1, −0.013 Dice). Its
encoder does, however, adapt slightly better to smartphone clinical images
when fine-tuned on them under the C1 protocol (+0.027 macro-F1 on PAD-UFES,
3/3 seeds). This is a seen-source result, not unseen-source generalisation.

## 4. Context (not comparable; spec §6)

- C (HAM only) macro-F1 0.68 vs production CV-4 0.5768: different training
  data (HAM subset vs all of ISIC 2019) and test set.
- S on ISIC 2018 test, Dice 0.8865, vs production CV-3 U-Net 0.8640: S was
  trained on HAM with Tschandl masks, and CV-3 on ISIC 2018. This is not the
  Experiment 2 CV-3 comparison, which trains the ResNet-50 U-Net on ISIC 2018
  itself.

## 5. Limitations

1. **One loss weight.** λ = 1 was fixed and not swept, by design. This result
   says "negative transfer at λ = 1", not "joint training cannot work".
2. **Three seeds.** The rule is a consistency check, not a significance test.
   The classification gap rests mostly on seeds 42 and 43.
3. **One epoch budget (30) for all arms.** C peaked earliest (epochs 9–21),
   and S and J later (20–29). Best-on-val selection covers early peaks, but S
   and J may not have fully converged at 30.
4. **Single-annotator masks.** The Tschandl masks come from one
   dermatologist, with no inter-rater agreement, so the HAM Dice ceiling is
   unknown.
5. **Squash resize.** 600×450 images and masks are resized to 512×512
   without keeping the aspect ratio. This is the same for all arms.
6. **PAD transfer resolution mismatch.** The encoders were trained at 512 and
   fine-tuned at 224, per the unchanged C1 protocol. This is the same for C
   and J.

## 6. What happens next

- The joint-model track stops (outcome C). No λ sweep, no GradNorm or
  uncertainty weighting, no extra seeds.
- The iToBoS partial-label spec is **not** unlocked.
- The PAD-UFES observation is **not** followed up under this spec. If it is
  ever pursued, it needs its own spec with a question, sample and decision
  rule written before any run, and it must not be framed as a rescue of this
  result.
- Experiment 2 (`docs/academic_resnet50_three_task_spec.md`) is independent
  of this outcome and can proceed. It reuses `src/academic/model.py` (arm S)
  for its CV-3 run.
