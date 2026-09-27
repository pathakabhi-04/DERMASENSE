#!/usr/bin/env bash
# All of docs/academic_vit_spec.md on the pod: GPU preflight, CV-4 ViT
# training + test, CV-3 ViT training + test, the two attention evaluations,
# and the collected results. Safe to rerun: finished steps are skipped, and
# training resumes from last.pt.
#
#   bash scripts/academic_vit/run_all.sh /workspace/dermasense_academic
set -euo pipefail

BASE=${1:?base dir, e.g. /workspace/dermasense_academic}
VIT=$BASE/data/academic_vit_bundle
R50=$BASE/data/academic_r50_bundle
HAM=$BASE/data/academic_joint_bundle
RUNS=$BASE/runs/academic_vit
export PYTHONPATH=.
mkdir -p "$RUNS"

[[ -f "$RUNS/preflight_gpu.json" ]] || \
  python3 scripts/academic_vit/preflight.py --out "$RUNS/preflight_gpu.json" 2>&1 | tee -a "$RUNS/preflight.log"

# CV-4: ViT-B/16 on ISIC 2019 (spec 3.1)
python3 scripts/academic_vit/train_cv4.py --data-root "$VIT" --run-dir "$RUNS/cv4" 2>&1 | tee -a "$RUNS/cv4_train.log"
[[ -f "$RUNS/cv4/test_metrics.json" ]] || \
  python3 scripts/academic_vit/evaluate_cv4.py --data-root "$VIT" --run-dir "$RUNS/cv4" 2>&1 | tee -a "$RUNS/cv4_train.log"

# CV-3: ViT-B/16 encoder + pyramid + unchanged U-Net decoder (Experiment 2 CV-3 recipe + the ViT lr change)
python3 scripts/academic_joint/train.py --arm S --seed 42 --dataset isic2018 --augment none --backbone vit_b_16 \
  --epochs 50 --batch-size 8 --weight-decay 0.01 --learning-rate 1e-4 --encoder-lr 3e-5 --warmup-epochs 1 \
  --data-root "$R50" --run-root "$RUNS" 2>&1 | tee -a "$RUNS/cv3_train.log"
[[ -f "$RUNS/S_seed42/test_metrics.json" ]] || \
  python3 scripts/academic_r50/evaluate_cv3.py --data-root "$R50" --run-dir "$RUNS/S_seed42" 2>&1 | tee -a "$RUNS/cv3_train.log"

# Attention maps (rollout + Grad-CAM)
[[ -f "$RUNS/attention/cv4_gradcam/metrics.json" ]] || \
  python3 scripts/academic_vit/attention.py --task cv4 --ham-root "$HAM" --run-root "$RUNS" 2>&1 | tee -a "$RUNS/attention_cv4.log"
[[ -f "$RUNS/attention/cv3_gradcam/metrics.json" ]] || \
  python3 scripts/academic_vit/attention.py --task cv3 --r50-root "$R50" --run-root "$RUNS" 2>&1 | tee -a "$RUNS/attention_cv3.log"

python3 scripts/academic_vit/collect.py --run-root "$RUNS" --out-dir "$RUNS/results" \
  --r50b evaluation/academic_vit/r50_on_bundle.json \
  --r50-cv3-per-image "$BASE/runs/academic_r50/S_seed42/test_isic2018_per_image.csv" \
  --r50-attention evaluation/academic_r50/attention_metrics.csv
echo "ALL STEPS COMPLETE"
