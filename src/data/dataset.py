from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

from src.data.manifest import Manifest, ManifestError
from src.data.targets import NativeTargetSpace


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_ROOTS = {
    "pad_ufes": PROJECT_ROOT / "data/splits/pad_ufes",
    "isic2019": PROJECT_ROOT / "data/splits/isic2019",
}

VALID_SPLITS = {"train", "val", "test"}


class DatasetError(RuntimeError):
    """Raised when a frozen CV split violates the runtime contract."""


@dataclass(frozen=True)
class CVSample:
    """
    Normalized runtime representation of one sample from a frozen split.
    """

    dataset_id: str
    split: str

    image_id: str
    image_path: Path

    native_diagnosis: str
    target_index: int
    image_domain: str

    evaluation_eligible: bool

    patient_id: Optional[str] = None
    lesion_id: Optional[str] = None
    operational_lesion_uid: Optional[str] = None
    label_strength: Optional[str] = None


class CVDataset:
    """
    Read-only runtime interface over a frozen CV split.

    Responsibilities:
      - load an already-frozen split;
      - validate its schema;
      - resolve physical image paths;
      - verify split membership;
      - expose normalized CVSample objects;
      - expose deterministic native diagnostic targets.

    This class does NOT:
      - create splits;
      - shuffle membership;
      - modify manifests;
      - alter labels;
      - map native diagnoses to risk categories.
    """

    REQUIRED_COMMON_COLUMNS = {
        "native_diagnosis",
        "image_domain",
        "split",
    }

    REQUIRED_COLUMNS = {
        "pad_ufes": {
            "dataset",
            "image_id",
            "patient_id",
            "lesion_id",
            "lesion_uid",
            "image_path",
            "native_diagnosis",
            "label_strength",
            "image_domain",
            "split",
        },
        "isic2019": {
            "image",
            "native_diagnosis",
            "lesion_id",
            "lesion_id_status",
            "operational_lesion_uid",
            "archive_path",
            "physical_filename",
            "image_domain",
            "label_strength",
            "split",
            "identity_status",
            "evaluation_eligible",
        },
    }

    def __init__(
        self,
        dataset_id: str,
        split: str,
        verify_images: bool = True,
    ) -> None:

        dataset_id = dataset_id.lower().strip()
        split = split.lower().strip()

        if dataset_id not in SPLIT_ROOTS:
            raise DatasetError(
                f"Unsupported dataset: {dataset_id!r}. "
                f"Expected: {sorted(SPLIT_ROOTS)}"
            )

        if split not in VALID_SPLITS:
            raise DatasetError(
                f"Unsupported split: {split!r}. "
                f"Expected: {sorted(VALID_SPLITS)}"
            )

        self.dataset_id = dataset_id
        self.split = split
        self.verify_images = verify_images

        self.split_path = (
            SPLIT_ROOTS[dataset_id] / f"{split}.csv"
        )

        if not self.split_path.exists():
            raise FileNotFoundError(
                f"Frozen split not found: {self.split_path}"
            )

        # Frozen manifest is the authoritative dataset membership source.
        self.manifest = Manifest(dataset_id)

        # Frozen native diagnostic target space.
        #
        # This intentionally remains dataset-specific.
        # It does NOT define LOW/MEDIUM/HIGH risk categories.
        self.target_space = NativeTargetSpace(dataset_id)

        self._df = pd.read_csv(self.split_path)

        self._validate_schema()
        self._validate_split_identity()
        self._validate_image_identity()
        self._validate_against_manifest()
        self._validate_evaluation_policy()

        if self.verify_images:
            self._validate_image_paths()

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    @property
    def num_samples(self) -> int:
        return len(self._df)

    @property
    def image_ids(self) -> set[str]:
        return set(
            self._df[self._image_id_column()].astype(str)
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _validate_schema(self) -> None:
        required = set(self.REQUIRED_COMMON_COLUMNS)
        required.update(
            self.REQUIRED_COLUMNS[self.dataset_id]
        )

        missing = sorted(
            required - set(self._df.columns)
        )

        if missing:
            raise DatasetError(
                f"{self.dataset_id}/{self.split} is missing "
                f"required columns: {missing}"
            )

    # ------------------------------------------------------------------
    # Split identity
    # ------------------------------------------------------------------

    def _validate_split_identity(self) -> None:
        split_values = (
            self._df["split"]
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
        )

        if len(split_values) != 1:
            raise DatasetError(
                f"{self.dataset_id}/{self.split} contains "
                f"multiple split values: "
                f"{split_values.tolist()}"
            )

        if split_values[0] != self.split:
            raise DatasetError(
                f"Split mismatch: requested {self.split!r}, "
                f"but file contains {split_values[0]!r}"
            )

        if "dataset" in self._df.columns:
            dataset_values = (
                self._df["dataset"]
                .dropna()
                .astype(str)
                .str.lower()
                .unique()
            )

            if (
                len(dataset_values) != 1
                or dataset_values[0] != self.dataset_id
            ):
                raise DatasetError(
                    f"Dataset mismatch in split. "
                    f"Expected {self.dataset_id!r}, "
                    f"found {dataset_values.tolist()}"
                )

    # ------------------------------------------------------------------
    # Image identity
    # ------------------------------------------------------------------

    def _image_id_column(self) -> str:
        if self.dataset_id == "pad_ufes":
            return "image_id"

        return "image"

    def _validate_image_identity(self) -> None:
        column = self._image_id_column()

        if self._df[column].isna().any():
            raise DatasetError(
                f"{self.dataset_id}/{self.split} contains "
                f"null image IDs."
            )

        ids = self._df[column].astype(str)

        if ids.duplicated().any():
            duplicates = (
                ids[ids.duplicated(keep=False)]
                .unique()
                .tolist()
            )

            raise DatasetError(
                f"{self.dataset_id}/{self.split} contains "
                f"duplicate image IDs: {duplicates[:10]}"
            )

    # ------------------------------------------------------------------
    # Manifest correspondence
    # ------------------------------------------------------------------

    def _validate_against_manifest(self) -> None:
        split_ids = self.image_ids
        manifest_ids = self.manifest.image_ids

        missing = split_ids - manifest_ids

        # "extra" is intentionally NOT an error because a split
        # represents only one partition of the complete manifest.

        if missing:
            raise DatasetError(
                f"{self.dataset_id}/{self.split} contains "
                f"{len(missing)} image IDs absent from the manifest. "
                f"Examples: {sorted(missing)[:10]}"
            )

        manifest_df = self.manifest.dataframe

        manifest_column = self.manifest._image_id_column()

        manifest_subset = manifest_df[
            manifest_df[manifest_column]
            .astype(str)
            .isin(split_ids)
        ].copy()

        if len(manifest_subset) != len(self._df):
            raise DatasetError(
                f"{self.dataset_id}/{self.split} does not have "
                f"a one-to-one correspondence with its manifest "
                f"subset."
            )

        self._validate_label_correspondence(
            manifest_subset
        )

    def _validate_label_correspondence(
        self,
        manifest_subset: pd.DataFrame,
    ) -> None:

        split_column = self._image_id_column()
        manifest_column = self.manifest._image_id_column()

        split_view = self._df[
            [split_column, "native_diagnosis", "image_domain"]
        ].copy()

        manifest_view = manifest_subset[
            [
                manifest_column,
                "native_diagnosis",
                "image_domain",
            ]
        ].copy()

        split_view[split_column] = (
            split_view[split_column].astype(str)
        )

        manifest_view[manifest_column] = (
            manifest_view[manifest_column].astype(str)
        )

        split_view = split_view.sort_values(
            split_column
        ).reset_index(drop=True)

        manifest_view = manifest_view.sort_values(
            manifest_column
        ).reset_index(drop=True)

        if not split_view[split_column].equals(
            manifest_view[manifest_column]
        ):
            raise DatasetError(
                f"{self.dataset_id}/{self.split} image "
                f"membership does not match manifest."
            )

        if not split_view["native_diagnosis"].equals(
            manifest_view["native_diagnosis"]
        ):
            raise DatasetError(
                f"{self.dataset_id}/{self.split} contains "
                f"native diagnosis values inconsistent with "
                f"the frozen manifest."
            )

        if not split_view["image_domain"].equals(
            manifest_view["image_domain"]
        ):
            raise DatasetError(
                f"{self.dataset_id}/{self.split} contains "
                f"image_domain values inconsistent with "
                f"the frozen manifest."
            )

    # ------------------------------------------------------------------
    # Evaluation policy
    # ------------------------------------------------------------------

    def _validate_evaluation_policy(self) -> None:
        if "evaluation_eligible" not in self._df.columns:
            return

        eligible = (
            self._df["evaluation_eligible"]
            .fillna(False)
            .astype(bool)
        )

        if self.dataset_id == "isic2019":
            unknown = (
                self._df["lesion_id_status"]
                .astype(str)
                .str.lower()
                == "unknown"
            )

            if unknown.any() and eligible[unknown].any():
                raise DatasetError(
                    f"ISIC 2019 {self.split} contains "
                    f"unknown-lesion-ID images marked "
                    f"evaluation eligible."
                )

            if self.split in {"val", "test"}:
                if not eligible.all():
                    raise DatasetError(
                        f"ISIC 2019 {self.split} contains "
                        f"evaluation-ineligible images."
                    )

            if self.split == "train":
                if eligible.any():
                    raise DatasetError(
                        "ISIC 2019 training split contains "
                        "evaluation-eligible images."
                    )

    # ------------------------------------------------------------------
    # Physical image resolution
    # ------------------------------------------------------------------

    def _resolve_image_path(
        self,
        row: pd.Series,
    ) -> Path:

        if self.dataset_id == "pad_ufes":
            relative_path = str(row["image_path"])

            path = PROJECT_ROOT / relative_path

        else:
            archive_path = str(row["archive_path"])

            path = (
                PROJECT_ROOT
                / "data/raw/isic2019"
                / archive_path
            )

        return path

    def _validate_image_paths(self) -> None:
        missing = []

        for _, row in self._df.iterrows():
            path = self._resolve_image_path(row)

            if not path.is_file():
                missing.append(str(path))

        if missing:
            examples = missing[:10]

            raise DatasetError(
                f"{self.dataset_id}/{self.split} contains "
                f"{len(missing)} missing physical images. "
                f"Examples: {examples}"
            )

    # ------------------------------------------------------------------
    # Sample conversion
    # ------------------------------------------------------------------

    def _optional_string(
        self,
        row: pd.Series,
        column: str,
    ) -> Optional[str]:

        if column not in row.index:
            return None

        value = row[column]

        if pd.isna(value):
            return None

        return str(value)

    def _row_to_sample(
        self,
        row: pd.Series,
    ) -> CVSample:

        image_column = self._image_id_column()

        if self.dataset_id == "pad_ufes":
            lesion_uid = self._optional_string(
                row,
                "lesion_uid",
            )

        else:
            lesion_uid = self._optional_string(
                row,
                "operational_lesion_uid",
            )

        # Evaluation eligibility is explicit for ISIC 2019.
        # PAD-UFES does not carry this column, so eligibility
        # is determined by the frozen split itself.
        if "evaluation_eligible" in row.index:
            value = row["evaluation_eligible"]

            if pd.isna(value):
                evaluation_eligible = False
            else:
                evaluation_eligible = bool(value)
        else:
            evaluation_eligible = self.split in {"val", "test"}

        # --------------------------------------------------------------
        # Native diagnostic target
        # --------------------------------------------------------------
        #
        # Keep the original diagnosis and additionally expose its
        # deterministic integer target.
        #
        # Example:
        #
        # PAD-UFES:
        #   SCC -> 4
        #
        # ISIC 2019:
        #   MEL -> 4
        #
        # These indices are dataset-specific and must never be
        # interpreted as a shared cross-dataset taxonomy.
        native_diagnosis = str(
            row["native_diagnosis"]
        )

        target_index = self.target_space.encode(
            native_diagnosis
        )

        return CVSample(
            dataset_id=self.dataset_id,
            split=self.split,

            image_id=str(row[image_column]),

            image_path=self._resolve_image_path(row),

            native_diagnosis=native_diagnosis,

            target_index=target_index,

            image_domain=str(
                row["image_domain"]
            ),

            evaluation_eligible=evaluation_eligible,

            patient_id=self._optional_string(
                row,
                "patient_id",
            ),

            lesion_id=self._optional_string(
                row,
                "lesion_id",
            ),

            operational_lesion_uid=lesion_uid,

            label_strength=self._optional_string(
                row,
                "label_strength",
            ),
        )

    # ------------------------------------------------------------------
    # Public sample access
    # ------------------------------------------------------------------

    def get(self, index: int) -> CVSample:
        if index < 0 or index >= len(self._df):
            raise IndexError(
                f"Sample index {index} out of range "
                f"for {self.dataset_id}/{self.split} "
                f"(size={len(self._df)})"
            )

        return self._row_to_sample(
            self._df.iloc[index]
        )

    def get_by_image_id(
        self,
        image_id: str,
    ) -> CVSample:

        image_id = str(image_id)

        column = self._image_id_column()

        matches = self._df[
            self._df[column].astype(str) == image_id
        ]

        if len(matches) == 0:
            raise KeyError(
                f"Image ID {image_id!r} not found in "
                f"{self.dataset_id}/{self.split}"
            )

        if len(matches) > 1:
            raise DatasetError(
                f"Image ID {image_id!r} occurs multiple "
                f"times in {self.dataset_id}/{self.split}"
            )

        return self._row_to_sample(
            matches.iloc[0]
        )

    def load_image(self, index: int) -> Image.Image:
        """
        Load one image as an RGB PIL image.
        """

        sample = self.get(index)

        if not sample.image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {sample.image_path}"
            )

        with Image.open(sample.image_path) as image:
            return image.convert("RGB")

    def records(self) -> list[CVSample]:
        return [
            self._row_to_sample(row)
            for _, row in self._df.iterrows()
        ]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> CVSample:
        return self.get(index)

    def __repr__(self) -> str:
        return (
            f"CVDataset("
            f"dataset_id={self.dataset_id!r}, "
            f"split={self.split!r}, "
            f"samples={self.num_samples}"
            f")"
        )