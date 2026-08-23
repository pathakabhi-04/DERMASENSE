from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATHS = {
    "pad_ufes": PROJECT_ROOT / "data/manifests/pad_ufes_manifest.csv",
    "isic2019": PROJECT_ROOT / "data/manifests/isic2019_manifest.csv",
}


@dataclass(frozen=True)
class ManifestRecord:
    """
    Normalized runtime representation of one image.

    The underlying frozen manifests have different schemas.
    This class provides one common interface without modifying
    the authoritative CSV artifacts.
    """

    dataset_id: str
    image_id: str
    native_diagnosis: str
    image_domain: str

    image_path: Optional[str] = None

    patient_id: Optional[str] = None
    lesion_id: Optional[str] = None
    operational_lesion_uid: Optional[str] = None

    label_strength: Optional[str] = None

    evaluation_eligible: bool = False
    split: Optional[str] = None


class ManifestError(RuntimeError):
    """Raised when a frozen manifest violates the runtime contract."""


class Manifest:
    """
    Read-only runtime interface over a frozen dataset manifest.

    This class does NOT:
    - create splits;
    - modify dataset membership;
    - alter labels;
    - rewrite manifest files.

    It only loads and validates an already-frozen manifest.
    """

    COMMON_REQUIRED_COLUMNS = {
        "native_diagnosis",
        "image_domain",
    }

    DATASET_REQUIRED_COLUMNS = {
        "pad_ufes": {
            "dataset",
            "image_id",
            "patient_id",
            "lesion_id",
            "lesion_uid",
            "image_path",
            "label_strength",
        },
        "isic2019": {
            "image",
            "lesion_id",
            "lesion_id_status",
            "operational_lesion_uid",
            "image_domain",
            "label_strength",
        },
    }

    def __init__(
        self,
        dataset_id: str,
        path: Optional[Path] = None,
    ) -> None:

        dataset_id = dataset_id.lower().strip()

        if dataset_id not in MANIFEST_PATHS:
            raise ManifestError(
                f"Unsupported dataset: {dataset_id!r}. "
                f"Expected one of: {sorted(MANIFEST_PATHS)}"
            )

        self.dataset_id = dataset_id
        self.path = Path(path) if path else MANIFEST_PATHS[dataset_id]

        if not self.path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.path}"
            )

        self._df = pd.read_csv(self.path)

        self._validate_schema()
        self._validate_dataset_identity()
        self._validate_identity()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Return a defensive copy.

        Callers cannot accidentally modify the internal manifest.
        """
        return self._df.copy()

    @property
    def columns(self) -> list[str]:
        return self._df.columns.tolist()

    @property
    def num_images(self) -> int:
        return len(self._df)

    @property
    def image_ids(self) -> set[str]:
        return set(
            self._df[self._image_id_column()]
            .astype(str)
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_schema(self) -> None:
        required = set(self.COMMON_REQUIRED_COLUMNS)

        required.update(
            self.DATASET_REQUIRED_COLUMNS[self.dataset_id]
        )

        missing = sorted(required - set(self._df.columns))

        if missing:
            raise ManifestError(
                f"{self.dataset_id} manifest is missing "
                f"required columns: {missing}"
            )

    def _validate_dataset_identity(self) -> None:
        """
        Ensure the dataset column, when present, agrees with the
        runtime dataset being loaded.
        """

        if "dataset" not in self._df.columns:
            return

        values = (
            self._df["dataset"]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
        )

        if len(values) != 1 or values[0] != self.dataset_id:
            raise ManifestError(
                f"Dataset identity mismatch. "
                f"Expected {self.dataset_id!r}, "
                f"found {values.tolist()}"
            )

    def _validate_identity(self) -> None:
        image_column = self._image_id_column()

        if self._df[image_column].isna().any():
            raise ManifestError(
                f"{self.dataset_id} manifest contains "
                f"null image IDs."
            )

        image_ids = self._df[image_column].astype(str)

        if image_ids.duplicated().any():
            duplicates = (
                image_ids[
                    image_ids.duplicated(keep=False)
                ]
                .unique()
                .tolist()
            )

            raise ManifestError(
                f"{self.dataset_id} manifest contains "
                f"duplicate image IDs: {duplicates[:10]}"
            )

        if self._df["native_diagnosis"].isna().any():
            raise ManifestError(
                f"{self.dataset_id} manifest contains images "
                f"without native diagnoses."
            )

        if self._df["image_domain"].isna().any():
            raise ManifestError(
                f"{self.dataset_id} manifest contains images "
                f"without image_domain."
            )

    # ------------------------------------------------------------------
    # Dataset-specific column normalization
    # ------------------------------------------------------------------

    def _image_id_column(self) -> str:
        if self.dataset_id == "pad_ufes":
            return "image_id"

        if self.dataset_id == "isic2019":
            return "image"

        raise ManifestError(
            f"Unknown dataset: {self.dataset_id}"
        )

    def _lesion_uid_column(self) -> str:
        if self.dataset_id == "pad_ufes":
            return "lesion_uid"

        if self.dataset_id == "isic2019":
            return "operational_lesion_uid"

        raise ManifestError(
            f"Unknown dataset: {self.dataset_id}"
        )

    # ------------------------------------------------------------------
    # Record access
    # ------------------------------------------------------------------

    def get(self, image_id: str) -> ManifestRecord:
        """
        Retrieve one immutable ManifestRecord by image ID.
        """

        image_id = str(image_id)
        image_column = self._image_id_column()

        matches = self._df[
            self._df[image_column].astype(str) == image_id
        ]

        if len(matches) == 0:
            raise KeyError(
                f"Image ID {image_id!r} not found in "
                f"{self.dataset_id} manifest."
            )

        if len(matches) > 1:
            raise ManifestError(
                f"Image ID {image_id!r} appears multiple times "
                f"in {self.dataset_id} manifest."
            )

        return self._row_to_record(matches.iloc[0])

    def _row_to_record(
        self,
        row: pd.Series,
    ) -> ManifestRecord:

        image_column = self._image_id_column()
        lesion_uid_column = self._lesion_uid_column()

        return ManifestRecord(
            dataset_id=self.dataset_id,

            image_id=str(row[image_column]),

            native_diagnosis=str(
                row["native_diagnosis"]
            ),

            image_domain=str(
                row["image_domain"]
            ),

            image_path=self._optional_string(
                row,
                "image_path",
            ),

            patient_id=self._optional_string(
                row,
                "patient_id",
            ),

            lesion_id=self._optional_string(
                row,
                "lesion_id",
            ),

            operational_lesion_uid=self._optional_string(
                row,
                lesion_uid_column,
            ),

            label_strength=self._optional_string(
                row,
                "label_strength",
            ),
        )

    @staticmethod
    def _optional_string(
        row: pd.Series,
        column: str,
    ) -> Optional[str]:

        if column not in row.index:
            return None

        value = row[column]

        if pd.isna(value):
            return None

        return str(value)

    # ------------------------------------------------------------------
    # Dataset inspection
    # ------------------------------------------------------------------

    def records(self) -> list[ManifestRecord]:
        """
        Return all normalized records.

        Intended for validation and inspection.
        """

        return [
            self._row_to_record(row)
            for _, row in self._df.iterrows()
        ]

    def diagnosis_counts(self) -> pd.Series:
        return self._df[
            "native_diagnosis"
        ].value_counts()

    def domain_counts(self) -> pd.Series:
        return self._df[
            "image_domain"
        ].value_counts()

    def has_column(self, column: str) -> bool:
        return column in self._df.columns

    def __len__(self) -> int:
        return self.num_images

    def __repr__(self) -> str:
        return (
            f"Manifest("
            f"dataset_id={self.dataset_id!r}, "
            f"images={self.num_images}, "
            f"path={str(self.path)!r}"
            f")"
        )