#!/usr/bin/env bash
# All 9 runs of docs/academic_joint_seg_cls_spec.md, then test evaluation and
# the PAD-UFES C1 transfer for C and J. Seed-major order, so each finished
# seed gives a complete (S, C, J) triple. Safe to rerun after a crash: finished
# training runs are skipped (DONE.json), unfinished ones resume from last.pt,
# and finished evaluations/transfers are skipped.
#
#   bash scripts/academic_joint/run_all.sh \
#     /workspace/dermasense_academic/data/academic_joint_bundle \
#     /workspace/dermasense_academic/runs/academic_joint
set -euo pipefail

DATA_ROOT=${1:?data root}
RUN_ROOT=${2:?run root}
export PYTHONPATH=.

mkdir -p "$RUN_ROOT"
for seed in 42 43 44; do
  for arm in S C J; do
    run="$RUN_ROOT/${arm}_seed${seed}"
    python3 scripts/academic_joint/train.py --arm "$arm" --seed "$seed" \
      --data-root "$DATA_ROOT" --run-root "$RUN_ROOT" 2>&1 | tee -a "$RUN_ROOT/${arm}_seed${seed}.log"
    [[ -f "$run/test_metrics.json" ]] || \
      python3 scripts/academic_joint/evaluate.py --run-dir "$run" --data-root "$DATA_ROOT" \
        2>&1 | tee -a "$RUN_ROOT/${arm}_seed${seed}.log"
    if [[ "$arm" != S && ! -f "$run/pad_c1_metrics.json" ]]; then
      python3 scripts/academic_joint/transfer_pad_c1.py --run-dir "$run" --data-root "$DATA_ROOT" \
        2>&1 | tee -a "$RUN_ROOT/${arm}_seed${seed}.log"
    fi
  done
done

python3 scripts/academic_joint/collect_results.py --run-root "$RUN_ROOT" --out-dir "$RUN_ROOT/results"
echo "ALL RUNS COMPLETE"
