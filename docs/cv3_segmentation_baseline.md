# CV-3 Segmentation Baseline

## 1. Purpose

CV-3 is the lesion segmentation component of DermaSense.

The purpose of this experiment is to establish a reproducible baseline for binary lesion segmentation on the ISIC 2018 Task 1 dataset before conducting further model or training-method experiments.

The baseline architecture is a U-Net trained with a combined Binary Cross-Entropy and Dice loss.

---

## 2. Official Baseline Configuration

| Component | Configuration |
|---|---|
| Architecture | U-Net |
| Input channels | 3 |
| Output channels | 1 |
| Input resolution | 512 × 512 |
| Base channels | 32 |
| Loss | 0.5 BCE + 0.5 Dice |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Epochs | 50 |
| Random seed | 42 |
| Threshold | 0.5 |
| Dataset | ISIC 2018 Task 1 |
| Train images | 2074 |
| Validation images | 259 |
| Test images | 260 |

The model outputs raw segmentation logits. Sigmoid is applied during metric computation rather than inside the model.

---

## 3. Dataset and Evaluation Protocol

The ISIC 2018 Task 1 segmentation dataset contains 2,594 training images with corresponding binary lesion masks.

The fixed split used for CV-3 is:

- Train: 2074 images
- Validation: 259 images
- Test: 260 images

The test set is kept separate from model selection and is evaluated only after training decisions have been made.

The primary segmentation metrics are:

- Dice coefficient
- Intersection over Union (IoU)

The reported test results use a probability threshold of 0.5.

---

## 4. Why U-Net Was Selected

U-Net was selected as the CV-3 baseline because it provides a straightforward encoder-decoder segmentation architecture with skip connections between corresponding spatial resolutions.

This makes it an appropriate baseline for lesion segmentation because the task requires both:

1. semantic understanding of the lesion region, and
2. preservation of spatial detail at the lesion boundary.

The implementation is intentionally kept simple so that later experiments can be compared against a clear and reproducible reference point.

---

## 5. Loss Function Selection

The initial CV-3 loss was:

    0.5 BCE + 0.5 Dice

where BCE is Binary Cross-Entropy with logits and Dice loss is:

    Dice Loss = 1 - Dice Score

A loss ablation was conducted to determine whether either component could be removed without degrading segmentation performance.

Three configurations were trained under the same experimental conditions:

1. 0.5 BCE + 0.5 Dice
2. BCE only
3. Dice only

---

## 6. Loss Ablation Results

### Validation

| Loss | Best Epoch | Validation Dice | Validation IoU |
|---|---:|---:|---:|
| 0.5 BCE + 0.5 Dice | 50 | 0.8652 | 0.7863 |
| Dice only | 47 | 0.8588 | 0.7777 |
| BCE only | 49 | 0.8600 | 0.7803 |

The combined BCE + Dice objective achieved the highest validation Dice and IoU.

### Held-out Test Set

| Loss | Best Epoch | Test Dice | Test IoU | Median Dice |
|---|---:|---:|---:|---:|
| 0.5 BCE + 0.5 Dice | 50 | 0.8640 | 0.7851 | 0.9136 |
| Dice only | 47 | 0.8549 | 0.7737 | 0.9078 |
| BCE only | 49 | 0.8670 | 0.7902 | 0.9179 |

BCE-only produced the highest observed mean test Dice and IoU. However, the difference from the combined loss was small.

---

## 7. Paired Statistical Comparison

Because all three models were evaluated on the same 260 test images, the BCE-only model was compared directly against the combined BCE + Dice baseline on a per-image basis.

The observed mean Dice difference was:

    BCE - BCE+Dice = +0.002993

The median difference was:

    -0.000218

BCE-only achieved a higher Dice score on 128 images, while the combined BCE + Dice model achieved a higher Dice score on 131 images. There was one tie.

A paired bootstrap with 10,000 resamples produced a 95% confidence interval of:

    [-0.005752, +0.011799]

The interval includes zero.

Therefore, the observed +0.0030 mean Dice advantage of BCE-only is not sufficient to establish that BCE-only is superior to the combined objective on this test split.

---

## 8. Baseline Decision

The official CV-3 baseline is therefore retained as:

> **U-Net + 0.5 BCE + 0.5 Dice**

The decision is based on the following evidence:

1. The combined loss achieved the strongest validation performance.
2. Dice-only performed worse on both validation and test evaluation.
3. BCE-only was competitive and achieved a slightly higher observed test mean.
4. The BCE-only improvement over the combined loss was small.
5. The paired bootstrap confidence interval for the BCE-only improvement included zero.
6. The per-image comparison did not show a consistent advantage for BCE-only.

The BCE-only result should therefore be recorded as a competitive ablation result rather than adopted as the new official baseline.

---

## 9. Official CV-3 Results

The current official baseline test result is:

- Dice: **0.8640**
- IoU: **0.7851**
- Median Dice: **0.9136**
- Median IoU: **0.8410**
- Test images: **260**
- Checkpoint epoch: **50**

These values are associated with:

    checkpoints/cv3/best.pt

and the corresponding evaluation artifacts in:

    evaluation/cv3/

---

## 10. Future Experiments

Future CV-3 experiments should treat this configuration as the reference baseline.

Unless an experiment explicitly changes a particular component, the following should remain fixed:

- U-Net architecture
- 512 × 512 input resolution
- fixed train/validation/test split
- seed 42
- AdamW optimizer
- learning rate 1e-4
- batch size 8
- 50 training epochs
- threshold 0.5

This allows subsequent architectural, preprocessing, augmentation, or training experiments to be compared against the same reference system.
