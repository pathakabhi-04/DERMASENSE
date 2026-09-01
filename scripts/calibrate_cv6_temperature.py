"""
CV-6 temperature calibration.

Fits a single scalar temperature on PAD-UFES val (labeled, per
docs/cv6_uncertainty_spec.md) that minimizes ECE, via
src.uncertainty.calibration.fit_temperature. The result is a fixed
constant used as the default in src/inference/orchestrator.py -- fit
once here, not refit at pipeline construction time.

Usage:
    python -m scripts.calibrate_cv6_temperature --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.transforms import ImageTransformConfig, build_eval_transform
from src.inference.native import PAD_CLASSES, NativePredictor
from src.uncertainty.calibration import (
    expected_calibration_error,
    fit_temperature,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PAD_UFES_VAL = REPO_ROOT / "data/splits/pad_ufes/val.csv"
CLASSIFIER_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints/archive/pad_ufes_c1_partial_finetune_seed42_best.pt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit CV-6's temperature-scaling constant on PAD-UFES val."
    )
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    return p.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    classifier = NativePredictor.from_checkpoint(
        CLASSIFIER_CHECKPOINT, device=device
    )
    transform = build_eval_transform(ImageTransformConfig())
    class_index = {name: i for i, name in enumerate(PAD_CLASSES)}

    val = pd.read_csv(PAD_UFES_VAL)

    probs = []
    labels = []
    for _, row in val.iterrows():
        image_path = REPO_ROOT / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image)
        prediction = classifier.predict(tensor)
        probs.append(
            [prediction.probabilities[name] for name in PAD_CLASSES]
        )
        labels.append(class_index[str(row["native_diagnosis"]).strip().upper()])

    probs = np.array(probs)
    labels = np.array(labels)

    raw_ece, _ = expected_calibration_error(probs, labels)
    temperature = fit_temperature(probs, labels)

    from src.uncertainty.calibration import apply_temperature

    calibrated_probs = apply_temperature(probs, temperature)
    calibrated_ece, _ = expected_calibration_error(calibrated_probs, labels)

    print(f"n = {len(labels)}")
    print(f"raw ECE:        {raw_ece:.4f}")
    print(f"fitted T:       {temperature:.2f}")
    print(f"calibrated ECE: {calibrated_ece:.4f}")


if __name__ == "__main__":
    main()
