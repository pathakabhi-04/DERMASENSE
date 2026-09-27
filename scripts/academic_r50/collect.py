"""Collect Experiment 2 into performance.csv / attention_metrics.csv and apply
the Section 4 and 5.3 rules mechanically.

    PYTHONPATH=. python3 scripts/academic_r50/collect.py \
        --run-root runs/academic_r50 --out-dir evaluation/academic_r50 \
        --baseline-e evaluation/academic_r50/baseline_e_yolo11s/predictions.csv.gz \
        --split-csv data/splits/itobos_detection/val.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location("measure_cv2", Path(__file__).with_name("measure_cv2.py"))
measure_cv2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure_cv2)

CV2_RECALL_MIN = 0.78
CV2_BURDEN_MEDIAN_MAX = 1
CV2_BURDEN_P90_MAX = 2
CV3_DICE_MIN = 0.854
BASELINES = {"cv3_dice": 0.8640, "cv4_macro_f1": 0.5756}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--baseline-e", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    frcnn = measure_cv2.measure_file(a.run_root / "cv2/predictions.csv", a.split_csv)
    yolo = measure_cv2.measure_file(a.baseline_e, a.split_csv)
    cv3 = json.loads((a.run_root / "S_seed42/test_metrics.json").read_text())

    cv2_match = (frcnn["image_recall"] >= CV2_RECALL_MIN and frcnn["burden_median"] <= CV2_BURDEN_MEDIAN_MAX
                 and frcnn["burden_p90"] <= CV2_BURDEN_P90_MAX)
    cv3_match = cv3["dice_mean"] >= CV3_DICE_MIN
    outcome = "C" if not cv3_match else ("A" if cv2_match else "B")

    perf = pd.DataFrame([
        {"task": "CV-2", "metric": "image_recall", "resnet50": frcnn["image_recall"], "baseline": yolo["image_recall"],
         "threshold": f">= {CV2_RECALL_MIN}", "pass": frcnn["image_recall"] >= CV2_RECALL_MIN},
        {"task": "CV-2", "metric": "burden_median", "resnet50": frcnn["burden_median"], "baseline": yolo["burden_median"],
         "threshold": f"<= {CV2_BURDEN_MEDIAN_MAX}", "pass": frcnn["burden_median"] <= CV2_BURDEN_MEDIAN_MAX},
        {"task": "CV-2", "metric": "burden_p90", "resnet50": frcnn["burden_p90"], "baseline": yolo["burden_p90"],
         "threshold": f"<= {CV2_BURDEN_P90_MAX}", "pass": frcnn["burden_p90"] <= CV2_BURDEN_P90_MAX},
        {"task": "CV-2", "metric": "box_recall (secondary)", "resnet50": frcnn["box_recall"], "baseline": yolo["box_recall"]},
        {"task": "CV-2", "metric": "binary_zero_fpr (secondary)", "resnet50": frcnn["binary_zero_fpr"], "baseline": yolo["binary_zero_fpr"]},
        {"task": "CV-2", "metric": "burden_p90, all 349 zero-lesion images (secondary)", "resnet50": frcnn["burden_p90_all"], "baseline": yolo["burden_p90_all"]},
        {"task": "CV-2", "metric": "burden_median, all 349 zero-lesion images (secondary)", "resnet50": frcnn["burden_median_all"], "baseline": yolo["burden_median_all"]},
        {"task": "CV-3", "metric": "test_dice", "resnet50": cv3["dice_mean"], "baseline": BASELINES["cv3_dice"],
         "threshold": f">= {CV3_DICE_MIN}", "pass": cv3_match},
        {"task": "CV-3", "metric": "test_iou (secondary)", "resnet50": cv3["iou_mean"], "baseline": 0.7851},
        {"task": "CV-4", "metric": "test_macro_f1", "resnet50": BASELINES["cv4_macro_f1"], "baseline": BASELINES["cv4_macro_f1"],
         "threshold": "by construction", "pass": True},
    ])
    perf.to_csv(a.out_dir / "performance.csv", index=False)

    att = []
    for task in ("cv2", "cv3", "cv4"):
        m = json.loads((a.run_root / f"attention/{task}/metrics.json").read_text())
        att.append({k: v for k, v in m.items() if k != "empty_prediction_ids"})
        shutil.copyfile(a.run_root / f"attention/{task}/attention_{task}.jpg", a.out_dir / f"attention_{task}.jpg")
    pd.DataFrame(att).to_csv(a.out_dir / "attention_metrics.csv", index=False)

    decision = {"cv2_matches": cv2_match, "cv3_matches": cv3_match, "outcome": outcome,
                "cv2_recall_ge_0.90_notable": frcnn["image_recall"] >= 0.90,
                "claims": {r["task"]: r["claim_allowed"] for r in att}}
    (a.out_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    print(perf.to_string(index=False))
    print(pd.DataFrame(att).to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
