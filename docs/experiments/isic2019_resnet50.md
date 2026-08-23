# ISIC 2019 — ResNet-50

## Dataset

- Dataset: ISIC 2019
- Native diagnosis classes: 8
- Train samples: 18,402
- Validation samples: 3,375
- Test samples: 3,554
- Split policy: lesion-disjoint
- Patient-level independence: NOT mechanically verifiable because `patient_id` is absent from ISIC 2019 metadata.

## Model

- Backbone: ResNet-50
- Pretrained: ImageNet
- Dropout: 0.0
- Total parameters: 23,536,718

### Parameter breakdown

| Component | Parameters |
|---|---:|
| Backbone | 23,508,032 |
| PAD-UFES head | 12,294 |
| ISIC 2019 head | 16,392 |
| Total | 23,536,718 |

## Training

- Epochs: 10
- Batch size: 32
- Learning rate: 0.0001
- Weight decay: 0.0001
- Optimizer: AdamW
- Scheduler: none
- Gradient clipping: none
- Primary metric: Macro-F1
- Test used during training: false

## Model Selection

- Best epoch: 7
- Best validation Macro-F1: 0.5864
- Checkpoint: `checkpoints/isic2019_resnet50_best.pt`

The checkpoint was selected using validation Macro-F1 only.

## Frozen Test Evaluation

- Accuracy: 0.734102
- Macro-F1: 0.578219
- Weighted-F1: 0.729576
- Loss: 0.959601

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| AK | 0.3387 | 0.2897 | 0.3123 | 145 |
| BCC | 0.7560 | 0.8147 | 0.7843 | 502 |
| BKL | 0.5011 | 0.6588 | 0.5692 | 337 |
| DF | 1.0000 | 0.3250 | 0.4906 | 40 |
| MEL | 0.6486 | 0.5405 | 0.5897 | 666 |
| NV | 0.8558 | 0.8792 | 0.8673 | 1714 |
| SCC | 0.3605 | 0.2844 | 0.3179 | 109 |
| VASC | 0.8065 | 0.6098 | 0.6944 | 41 |

## Comparison Against ResNet-18

| Metric | ResNet-18 | ResNet-50 | Change |
|---|---:|---:|---:|
| Accuracy | 0.717501 | 0.734102 | +0.016601 |
| Macro-F1 | 0.522276 | 0.578219 | +0.055943 |
| Weighted-F1 | 0.706109 | 0.729576 | +0.023467 |
| Best Val Macro-F1 | 0.5445 | 0.5864 | +0.0419 |

### Per-class F1 change

| Class | ResNet-18 | ResNet-50 | Change |
|---|---:|---:|---:|
| AK | 0.3684 | 0.3123 | -0.0561 |
| BCC | 0.7443 | 0.7843 | +0.0400 |
| BKL | 0.4740 | 0.5692 | +0.0952 |
| DF | 0.2759 | 0.4906 | +0.2147 |
| MEL | 0.5794 | 0.5897 | +0.0103 |
| NV | 0.8603 | 0.8673 | +0.0070 |
| SCC | 0.2267 | 0.3179 | +0.0912 |
| VASC | 0.6494 | 0.6944 | +0.0450 |

## Interpretation

ResNet-50 substantially improves overall Macro-F1 over ResNet-18.

The frozen test Macro-F1 improves from 0.522276 to 0.578219.

The largest class-level improvements are observed for:

- DF: +0.2147 F1
- BKL: +0.0952 F1
- SCC: +0.0912 F1

AK remains weak and decreases relative to ResNet-18.

## Status

Current best ISIC 2019 model.

Test evaluation is frozen and must not be used for subsequent model selection.
