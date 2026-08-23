# ISIC 2019 — ResNet-18 Baseline

## Dataset

- Dataset: ISIC 2019
- Native diagnosis classes: 8
- Train samples: 18,402
- Validation samples: 3,375
- Test samples: 3,554
- Split policy: lesion-disjoint
- Patient-level independence: NOT mechanically verifiable because `patient_id` is absent from ISIC 2019 metadata.

## Model

- Backbone: ResNet-18
- Pretrained: ImageNet
- Dropout: 0.0
- Total parameters: 11,183,694

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

- Best epoch: 10
- Best validation Macro-F1: 0.5445
- Checkpoint: `checkpoints/isic2019_resnet18_best.pt`

## Frozen Test Evaluation

- Accuracy: 0.717501
- Macro-F1: 0.522276
- Weighted-F1: 0.706109
- Loss: 1.083921

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| AK | 0.4050 | 0.3379 | 0.3684 | 145 |
| BCC | 0.7143 | 0.7769 | 0.7443 | 502 |
| BKL | 0.4890 | 0.4599 | 0.4740 | 337 |
| DF | 0.4444 | 0.2000 | 0.2759 | 40 |
| MEL | 0.5949 | 0.5646 | 0.5794 | 666 |
| NV | 0.8302 | 0.8926 | 0.8603 | 1714 |
| SCC | 0.4146 | 0.1560 | 0.2267 | 109 |
| VASC | 0.6944 | 0.6098 | 0.6494 | 41 |

## Status

Frozen baseline. Test results must not be used for subsequent model selection.
