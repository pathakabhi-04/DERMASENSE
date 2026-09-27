#!/usr/bin/env bash
# All of docs/academic_resnet50_three_task_spec.md on the pod: CV-3 training
# (short) first, then CV-2 training (long), the prediction export, the three
# attention evaluations and the collected results. Safe to rerun after a
# crash: finished steps are skipped, and training resumes from last.pt.
#
#   bash scripts/academic_r50/run_all.sh /workspace/dermasense_academic
set -euo pipefail

BASE=${1:?base dir, e.g. /workspace/dermasense_academic}
R50=$BASE/data/academic_r50_bundle
HAM=$BASE/data/academic_joint_bundle
RUNS=$BASE/runs/academic_r50
export PYTHONPATH=.
mkdir -p "$RUNS"

# CV-3: ResNet-50 U-Net on ISIC 2018 Task 1, CV-3 baseline settings (spec 3.1)
python3 scripts/academic_joint/train.py --arm S --seed 42 --dataset isic2018 --augment none \
  --epochs 50 --batch-size 8 --weight-decay 0.01 \
  --data-root "$R50" --run-root "$RUNS" 2>&1 | tee -a "$RUNS/cv3_train.log"
[[ -f "$RUNS/S_seed42/test_metrics.json" ]] || \
  python3 scripts/academic_r50/evaluate_cv3.py --data-root "$R50" --run-dir "$RUNS/S_seed42" 2>&1 | tee -a "$RUNS/cv3_train.log"

# CV-2: Faster R-CNN on iToBoS
python3 scripts/academic_r50/train_cv2.py --data-root "$R50" --run-dir "$RUNS/cv2" 2>&1 | tee -a "$RUNS/cv2_train.log"
[[ -f "$RUNS/cv2/predictions.csv" ]] || \
  python3 scripts/academic_r50/export_cv2_predictions.py --data-root "$R50" --run-dir "$RUNS/cv2" 2>&1 | tee -a "$RUNS/cv2_train.log"

# Attention maps
for task in cv4 cv3 cv2; do
  [[ -f "$RUNS/attention/$task/metrics.json" ]] || \
    python3 scripts/academic_r50/attention.py --task "$task" --r50-root "$R50" --ham-root "$HAM" \
      --run-root "$RUNS" 2>&1 | tee -a "$RUNS/attention_$task.log"
done

python3 scripts/academic_r50/collect.py --run-root "$RUNS" --out-dir "$RUNS/results" \
  --baseline-e evaluation/academic_r50/baseline_e_yolo11s/predictions.csv.gz \
  --split-csv data/splits/itobos_detection/val.csv
echo "ALL STEPS COMPLETE"
