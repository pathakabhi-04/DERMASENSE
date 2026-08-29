from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoundingBox:
    """A single normalized YOLO bounding box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.class_id != 0:
            raise ValueError(
                f"Expected lesion class_id=0, got {self.class_id}"
            )

        values = (
            self.x_center,
            self.y_center,
            self.width,
            self.height,
        )

        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(
                f"Bounding-box coordinates must be normalized to [0, 1]: "
                f"{values}"
            )

        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError(
                "Bounding-box width and height must be > 0."
            )


@dataclass(frozen=True)
class DetectionSample:
    """One iToBoS image and its lesion annotations."""

    image_id: str
    image_path: Path
    label_path: Path
    boxes: tuple[BoundingBox, ...]

    @property
    def has_lesions(self) -> bool:
        return len(self.boxes) > 0

    @property
    def num_lesions(self) -> int:
        return len(self.boxes)

