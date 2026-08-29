from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_WEIGHTS = "yolo11n.pt"
NUM_CLASSES = 1
CLASS_NAMES = ["lesion"]


def build_detector(
    weights: str | Path = DEFAULT_WEIGHTS,
    *,
    pretrained: bool = True,
    **kwargs: Any,
):
    """
    Build the CV-2 lesion candidate detector.

    CV-2 is a single-class object detector:
        class 0 -> lesion

    Parameters
    ----------
    weights:
        YOLO checkpoint or model configuration.

    pretrained:
        Whether the supplied weights should be treated as
        pretrained weights.

    kwargs:
        Additional arguments reserved for the detector
        implementation.

    Returns
    -------
    Detector model instance.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for the CV-2 detector. "
            "Install it with: pip install ultralytics"
        ) from exc

    weights = str(weights)

    if pretrained:
        model = YOLO(weights)
    else:
        model = YOLO(weights)

    return model


def detector_class_names() -> list[str]:
    """Return the fixed CV-2 class vocabulary."""
    return CLASS_NAMES.copy()


def detector_num_classes() -> int:
    """Return the number of CV-2 detection classes."""
    return NUM_CLASSES
