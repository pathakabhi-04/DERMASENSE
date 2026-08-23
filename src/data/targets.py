from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class TargetError(ValueError):
    """Raised when a diagnosis cannot be represented by the target space."""


# ----------------------------------------------------------------------
# Frozen native diagnostic spaces
# ----------------------------------------------------------------------

PAD_UFES_CLASSES = (
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
)

ISIC2019_CLASSES = (
    "AK",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "SCC",
    "VASC",
)


NATIVE_CLASS_SPACES: Mapping[str, tuple[str, ...]] = {
    "pad_ufes": PAD_UFES_CLASSES,
    "isic2019": ISIC2019_CLASSES,
}


@dataclass(frozen=True)
class NativeTarget:
    """
    Model target for one native diagnostic label.

    risk_category is intentionally absent.

    Risk/action reasoning belongs to the downstream risk layer,
    not the dataset target layer.
    """

    dataset_id: str
    diagnosis: str
    class_index: int


class NativeTargetSpace:
    """
    Dataset-specific native diagnostic target space.

    This class provides deterministic:
        diagnosis -> class index
        class index -> diagnosis

    It does NOT:
        - collapse diagnoses into risk categories;
        - combine the two datasets into one taxonomy;
        - infer clinical severity;
        - modify native labels.
    """

    def __init__(self, dataset_id: str) -> None:
        dataset_id = dataset_id.lower().strip()

        if dataset_id not in NATIVE_CLASS_SPACES:
            raise TargetError(
                f"Unsupported dataset: {dataset_id!r}. "
                f"Expected one of: "
                f"{sorted(NATIVE_CLASS_SPACES)}"
            )

        self.dataset_id = dataset_id
        self.classes = NATIVE_CLASS_SPACES[dataset_id]

        self._diagnosis_to_index = {
            diagnosis: index
            for index, diagnosis in enumerate(self.classes)
        }

        self._index_to_diagnosis = {
            index: diagnosis
            for index, diagnosis in enumerate(self.classes)
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.classes

    @property
    def diagnosis_to_index(self) -> dict[str, int]:
        return self._diagnosis_to_index.copy()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_diagnosis(self, diagnosis: str) -> None:
        diagnosis = str(diagnosis).strip()

        if diagnosis not in self._diagnosis_to_index:
            raise TargetError(
                f"Unknown native diagnosis {diagnosis!r} "
                f"for dataset {self.dataset_id!r}. "
                f"Allowed classes: {self.classes}"
            )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, diagnosis: str) -> int:
        diagnosis = str(diagnosis).strip()

        self.validate_diagnosis(diagnosis)

        return self._diagnosis_to_index[diagnosis]

    def encode_many(
        self,
        diagnoses: list[str],
    ) -> list[int]:
        return [
            self.encode(diagnosis)
            for diagnosis in diagnoses
        ]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, class_index: int) -> str:
        if class_index not in self._index_to_diagnosis:
            raise TargetError(
                f"Invalid class index {class_index} "
                f"for dataset {self.dataset_id!r}. "
                f"Valid range: "
                f"0..{self.num_classes - 1}"
            )

        return self._index_to_diagnosis[class_index]

    def decode_many(
        self,
        class_indices: list[int],
    ) -> list[str]:
        return [
            self.decode(index)
            for index in class_indices
        ]

    # ------------------------------------------------------------------
    # Target object
    # ------------------------------------------------------------------

    def target(self, diagnosis: str) -> NativeTarget:
        diagnosis = str(diagnosis).strip()

        return NativeTarget(
            dataset_id=self.dataset_id,
            diagnosis=diagnosis,
            class_index=self.encode(diagnosis),
        )

    # ------------------------------------------------------------------
    # Dataset-level validation
    # ------------------------------------------------------------------

    def validate_diagnoses(
        self,
        diagnoses: list[str],
    ) -> None:
        for diagnosis in diagnoses:
            self.validate_diagnosis(diagnosis)

    def __repr__(self) -> str:
        return (
            f"NativeTargetSpace("
            f"dataset_id={self.dataset_id!r}, "
            f"num_classes={self.num_classes}, "
            f"classes={self.classes!r}"
            f")"
        )