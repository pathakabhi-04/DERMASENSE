import json
from pathlib import Path
from typing import Any

from src.rag.ingestion.schema import MedicalDocument


class CorpusManifestLoader:
    """
    Loads and validates the DermaSense medical corpus manifest.
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)

    def load_manifest(self) -> dict[str, Any]:
        """Load the corpus manifest from JSON."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Corpus manifest not found: {self.manifest_path}"
            )

        with self.manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        self._validate_manifest(manifest)

        return manifest

    @staticmethod
    def _validate_manifest(manifest: dict[str, Any]) -> None:
        """Validate the minimum manifest structure."""

        required_fields = {
            "corpus_id",
            "version",
            "documents",
        }

        missing = required_fields - manifest.keys()

        if missing:
            raise ValueError(
                f"Manifest is missing required fields: {sorted(missing)}"
            )

        if not isinstance(manifest["documents"], list):
            raise TypeError("'documents' must be a list")

        for index, document in enumerate(manifest["documents"]):
            required_document_fields = {
                "document_id",
                "title",
                "source",
                "source_url",
                "topic",
            }

            missing_document_fields = (
                required_document_fields - document.keys()
            )

            if missing_document_fields:
                raise ValueError(
                    f"Document at index {index} is missing: "
                    f"{sorted(missing_document_fields)}"
                )

    def load_document_entries(self) -> list[dict[str, Any]]:
        """
        Return the raw document entries from the manifest.
        """
        manifest = self.load_manifest()
        return manifest["documents"]

    def build_document(
        self,
        entry: dict[str, Any],
        text: str,
    ) -> MedicalDocument:
        """
        Convert a manifest entry + extracted text into a MedicalDocument.
        """

        return MedicalDocument(
            document_id=entry["document_id"],
            title=entry["title"],
            source=entry["source"],
            source_url=entry.get("source_url"),
            text=text,
            topic=entry.get("topic"),
            condition=entry.get("condition"),
            sections=entry.get("sections", []),
            metadata={
                "corpus_id": self.load_manifest()["corpus_id"],
                "corpus_version": self.load_manifest()["version"],
            },
        )


class LocalTextLoader:
    """
    Loads plain-text medical source files from disk.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    def load(self, path: str | Path) -> str:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        return path.read_text(encoding="utf-8")