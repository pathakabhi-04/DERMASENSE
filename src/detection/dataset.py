from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import BoundingBox, DetectionSample


class ItobosDetectionDataset:
    """
    Dataset loader for the DermaSense CV-2 iToBoS detection dataset.

    The loader is intentionally model-independent. It reads a split
    manifest and returns validated DetectionSample objects.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path = "data/raw/itobos",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        self._df = pd.read_csv(self.manifest_path)

        required_columns = {
            "image_id",
            "image_path",
        }

        missing = required_columns - set(self._df.columns)

        if missing:
            raise ValueError(
                f"Manifest is missing required columns: "
                f"{sorted(missing)}"
            )

        if self._df["image_id"].duplicated().any():
            raise ValueError(
                "Manifest contains duplicate image IDs."
            )

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index: int) -> DetectionSample:
        row = self._df.iloc[index]

        image_id = str(row["image_id"])

        image_path = self._resolve_image_path(
            row["image_path"]
        )

        label_path = self._label_path(image_id)

        boxes = self._read_labels(label_path)

        return DetectionSample(
            image_id=image_id,
            image_path=image_path,
            label_path=label_path,
            boxes=tuple(boxes),
        )

    def _resolve_image_path(
        self,
        manifest_path: str,
    ) -> Path:
        path = Path(manifest_path)

        if path.is_absolute():
            resolved = path
        else:
            resolved = Path(manifest_path)

        if not resolved.exists():
            raise FileNotFoundError(
                f"Image not found: {resolved}"
            )

        return resolved

    def _label_path(
        self,
        image_id: str,
    ) -> Path:
        path = (
            self.data_root
            / "_train"
            / "_train"
            / "labels"
            / f"{image_id}.txt"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Label file not found for {image_id}: {path}"
            )

        return path

    @staticmethod
    def _read_labels(
        label_path: Path,
    ) -> list[BoundingBox]:
        boxes: list[BoundingBox] = []

        with label_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                # Empty label files represent zero-lesion images.
                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(
                        f"Malformed annotation in {label_path} "
                        f"at line {line_number}: {line!r}"
                    )

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid annotation in {label_path} "
                        f"at line {line_number}: {line!r}"
                    ) from exc

                boxes.append(
                    BoundingBox(
                        class_id=class_id,
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                    )
                )

        return boxes
