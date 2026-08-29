from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from .model import (
    CLASS_NAMES,
    NUM_CLASSES,
    build_detector,
    detector_class_names,
    detector_num_classes,
)


def test_class_configuration() -> None:
    assert NUM_CLASSES == 1
    assert CLASS_NAMES == ["lesion"]

    assert detector_num_classes() == 1
    assert detector_class_names() == ["lesion"]


def test_detector_construction() -> None:
    model = build_detector()

    assert model is not None
    assert hasattr(model, "names")

    # The pretrained checkpoint retains its original class vocabulary.
    # The CV-2 single-class configuration is applied during training
    # through the iToBoS dataset configuration.
    names = model.names

    assert names is not None
    assert len(names) > 0


def test_detector_inference() -> None:
    model = build_detector()

    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "test_image.jpg"

        image = Image.new(
            "RGB",
            (640, 640),
            color=(128, 128, 128),
        )
        image.save(image_path)

        results = model.predict(
            source=str(image_path),
            verbose=False,
        )

    assert len(results) == 1

    result = results[0]

    assert hasattr(result, "boxes")
    assert result.boxes is not None

    assert hasattr(result.boxes, "xyxy")
    assert hasattr(result.boxes, "conf")
    assert hasattr(result.boxes, "cls")


if __name__ == "__main__":
    test_class_configuration()
    test_detector_construction()
    test_detector_inference()

    print()
    print("=" * 80)
    print("CV-2 MODEL TESTS PASSED")
    print("=" * 80)