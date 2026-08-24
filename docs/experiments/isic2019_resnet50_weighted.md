# ISIC 2019 — ResNet-50 with Class Weighting

## Experiment

- Dataset: ISIC 2019
- Backbone: ResNet-50
- Pretrained: ImageNet
- Epochs: 10
- Batch size: 32
- Learning rate: 0.0001
- Optimizer: AdamW
- Weight decay: 0.0001
- Dropout: 0.0
- Seed: 42
- Primary metric: Macro-F1
- Class weighting: sqrt-inverse-frequency
- Architecture: `CV_MODEL_ARCHITECTURE_v1.0`

## Dataset

| Split | Samples |
|---|---:|
| Train | 18,402 |
| Validation | 3,375 |
| Test | 3,554 |

Classes:

`AK, BCC, BKL, DF, MEL, NV, SCC, VASC`

## Class Weights

| Class | Count | Weight |
|---|---:|---:|
| AK | 604 | 1.0347 |
| BCC | 2,343 | 0.5254 |
| BKL | 1,935 | 0.5781 |
| DF | 164 | 1.9857 |
| MEL | 3,256 | 0.4457 |
| NV | 9,512 | 0.2607 |
| SCC | 412 | 1.2528 |
| VASC | 176 | 1.9168 |

## Training Result

Best epoch: **6**

Best validation Macro-F1: **0.5963**

| Epoch | Train Loss | Train Macro-F1 | Val Loss | Val Macro-F1 |
|---|---:|---:|---:|---:|
| 1 | 1.1510 | 0.4673 | 1.0190 | 0.5472 |
| 2 | 0.7786 | 0.6258 | 0.9732 | 0.5683 |
| 3 | 0.6390 | 0.7061 | 1.0139 | 0.5757 |
| 4 | 0.5295 | 0.7572 | 1.0772 | 0.5625 |
| 5 | 0.4473 | 0.8026 | 1.1512 | 0.5833 |
| 6 | 0.3716 | 0.8328 | 1.1697 | **0.5963** |
| 7 | 0.3262 | 0.8569 | 1.2325 | 0.5898 |
| 8 | 0.2828 | 0.8830 | 1.2980 | 0.5883 |
| 9 | 0.2422 | 0.8999 | 1.4286 | 0.5630 |
| 10 | 0.2219 | 0.9038 | 1.3699 | 0.5796 |

The best checkpoint was saved at epoch 6.

## Frozen Test Evaluation

| Metric | Result |
|---|---:|
| Loss | 0.884760 |
| Accuracy | 0.729038 |
| Macro-F1 | **0.575637** |
| Weighted-F1 | 0.726834 |

### Confusion Matrix

Rows = true class, columns = predicted class.

| | AK | BCC | BKL | DF | MEL | NV | SCC | VASC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AK | 51 | 24 | 32 | 0 | 10 | 6 | 21 | 1 |
| BCC | 29 | 409 | 18 | 2 | 18 | 14 | 10 | 2 |
| BKL | 23 | 13 | 210 | 1 | 33 | 49 | 6 | 2 |
| DF | 3 | 8 | 12 | 14 | 1 | 1 | 1 | 0 |
| MEL | 9 | 30 | 69 | 4 | 377 | 167 | 10 | 0 |
| NV | 8 | 37 | 51 | 9 | 131 | 1467 | 4 | 7 |
| SCC | 12 | 33 | 18 | 0 | 6 | 5 | 34 | 1 |
| VASC | 0 | 7 | 1 | 0 | 4 | 0 | 0 | 29 |

### Per-Class Test Metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| AK | 0.3778 | 0.3517 | 0.3643 | 145 |
| BCC | 0.7291 | 0.8147 | 0.7695 | 502 |
| BKL | 0.5109 | 0.6231 | 0.5615 | 337 |
| DF | 0.4667 | 0.3500 | 0.4000 | 40 |
| MEL | 0.6500 | 0.5661 | 0.6051 | 666 |
| NV | 0.8584 | 0.8559 | 0.8571 | 1,714 |
| SCC | 0.3953 | 0.3119 | 0.3487 | 109 |
| VASC | 0.6905 | 0.7073 | 0.6988 | 41 |

## Comparison with Unweighted ResNet-50

| Metric | Unweighted | Weighted |
|---|---:|---:|
| Best Val Macro-F1 | 0.5864 | **0.5963** |
| Test Accuracy | 0.733821 | 0.729038 |
| Test Macro-F1 | **0.576770** | 0.575637 |
| Test Weighted-F1 | 0.729343 | 0.726834 |

Validation Macro-F1 improved by 0.0099 with class weighting, but frozen-test Macro-F1 changed from 0.576770 to 0.575637.

Therefore, class weighting did **not** demonstrate a meaningful improvement in held-out generalization in this experiment.

The weighting strategy improved F1 for AK, MEL, SCC, and VASC, but reduced F1 for BCC, BKL, DF, and NV on the frozen test set.

## Checkpoint

`checkpoints/isic2019_resnet50_weighted_best.pt`

The checkpoint was copied to the local machine and verified against the RunPod copy using SHA-256.

SHA-256:

`ef58fafd983727ad7e7936f192b5db061f353cbfbdb0086180dd5276e379598e`

## Conclusion

The weighted ResNet-50 achieved the highest validation Macro-F1 among the ResNet-50 experiments, but this improvement did not transfer to the frozen test set.

The unweighted ResNet-50 therefore remains the stronger test-set baseline:

**Test Macro-F1 = 0.5768.**

Class weighting should not be treated as an improvement based on this experiment alone.
