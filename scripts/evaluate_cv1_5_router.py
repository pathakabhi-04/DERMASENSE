"""
CV-1.5 router -- re-score an already-trained Stage 2 checkpoint.

Separate from scripts/train_cv1_5_router.py so a checkpoint that trained
successfully but couldn't be scored (e.g. the iToBoS test-split images
weren't available on the machine at the time) doesn't require burning
GPU time on a full retrain -- this just loads the checkpoint and re-runs
the final held-out evaluation against
analysis/quality/cv1_5_router/eval_set.csv (the same set Stage 1 and
Stage 2 training both use).

Usage:
    python -m scripts.evaluate_cv1_5_router --checkpoint checkpoints/cv1_5_router/best.pt --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.routing.classifier import load_router_checkpoint
from scripts.train_cv1_5_router import (
    PER_CLASS_GATE,
    RESULT_DIR,
    resolve_device,
    run_final_holdout_eval,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/cv1_5_router/best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-score a trained CV-1.5 router checkpoint against the held-out set."
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    model = load_router_checkpoint(str(args.checkpoint), device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    holdout_metrics, holdout_predictions = run_final_holdout_eval(
        model, device
    )
    per_class = holdout_metrics["per_class_accuracy"]
    pre_framed_acc = per_class["pre_framed"]
    wide_field_acc = per_class["wide_field"]
    passed = (
        pre_framed_acc >= PER_CLASS_GATE
        and wide_field_acc >= PER_CLASS_GATE
    )

    summary_lines = [
        "CV-1.5 Domain Router -- Stage 2 (Classifier) Result",
        "=" * 60,
        "",
        f"Checkpoint: {args.checkpoint} (re-scored, not retrained)",
        "",
        f"pre_framed accuracy: {pre_framed_acc:.3f}  (gate >= {PER_CLASS_GATE})",
        f"wide_field accuracy: {wide_field_acc:.3f}  (gate >= {PER_CLASS_GATE})",
        "",
        f"RESULT: {'PASS' if passed else 'FAIL'}",
    ]
    summary_text = "\n".join(summary_lines)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "stage2_summary.txt").write_text(summary_text + "\n")
    holdout_predictions.to_csv(
        RESULT_DIR / "stage2_predictions.csv", index=False
    )

    print()
    print(summary_text)
    print(f"\nSummary written to:     {RESULT_DIR / 'stage2_summary.txt'}")
    print(f"Predictions written to: {RESULT_DIR / 'stage2_predictions.csv'}")


if __name__ == "__main__":
    main()
