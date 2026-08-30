from __future__ import annotations

import hashlib
from pathlib import Path

from src.rag.ingestion.acquisition import load_manifest
from src.rag.ingestion.html_extractor import MedicalHTMLExtractor
from src.rag.ingestion.schema import MedicalDocument


class CorpusLoadError(RuntimeError):
    """Raised when an acquired medical document cannot be loaded safely."""


class MedicalCorpusLoader:
    """
    Convert successfully acquired source snapshots into MedicalDocument
    objects while preserving corpus and acquisition provenance.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        acquisition_manifest_path: str | Path,
        raw_dir: str | Path,
    ):
        self.manifest_path = Path(manifest_path)
        self.acquisition_manifest_path = Path(
            acquisition_manifest_path
        )
        self.raw_dir = Path(raw_dir)

        self.extractor = MedicalHTMLExtractor()

    def load(self) -> list[MedicalDocument]:
        """
        Load all successfully acquired documents from the corpus.

        Returns:
            List of MedicalDocument objects.
        """

        manifest = load_manifest(
            self.manifest_path
        )

        acquisition_records = load_manifest(
            self.acquisition_manifest_path
        )

        documents_by_id = {
            document["document_id"]: document
            for document in manifest["documents"]
        }

        documents: list[MedicalDocument] = []

        for record in acquisition_records:

            # Ignore unavailable / failed sources.
            if not record.get(
                "acquired",
                False,
            ):
                continue

            document_id = record["document_id"]

            if document_id not in documents_by_id:
                raise CorpusLoadError(
                    f"Document {document_id} exists in "
                    "acquisition manifest but not corpus manifest."
                )

            manifest_document = documents_by_id[
                document_id
            ]

            local_path = Path(
                record["local_path"]
            )

            # Protect against relative-path ambiguity.
            if not local_path.is_absolute():
                local_path = Path.cwd() / local_path

            if not local_path.exists():
                raise CorpusLoadError(
                    f"Acquired file does not exist: "
                    f"{local_path}"
                )

            raw_bytes = local_path.read_bytes()

            # Verify the downloaded snapshot has not changed.
            actual_sha256 = hashlib.sha256(
                raw_bytes
            ).hexdigest()

            expected_sha256 = record[
                "sha256"
            ]

            if actual_sha256 != expected_sha256:
                raise CorpusLoadError(
                    f"SHA-256 mismatch for {document_id}: "
                    f"expected {expected_sha256}, "
                    f"got {actual_sha256}"
                )

            raw_html = raw_bytes.decode(
                "utf-8",
                errors="replace",
            )

            try:
                text = self.extractor.extract(
                    raw_html
                )
            except Exception as exc:
                raise CorpusLoadError(
                    f"HTML extraction failed for "
                    f"{document_id}: {exc}"
                ) from exc

            if not text.strip():
                raise CorpusLoadError(
                    f"No meaningful medical text extracted "
                    f"from {document_id}"
                )

            document = MedicalDocument(
                document_id=document_id,
                title=manifest_document["title"],
                source=manifest_document["source"],
                source_url=manifest_document[
                    "source_url"
                ],
                text=text,
                topic=manifest_document["topic"],
                condition=manifest_document[
                    "condition"
                ],
                sections=manifest_document[
                    "sections"
                ],
                metadata={
                    "corpus_id": manifest[
                        "corpus_id"
                    ],
                    "corpus_version": manifest[
                        "version"
                    ],
                    "retrieved_at": record[
                        "retrieved_at"
                    ],
                    "sha256": expected_sha256,
                    "content_type": record[
                        "content_type"
                    ],
                    "byte_size": record[
                        "byte_size"
                    ],
                },
            )

            documents.append(
                document
            )

        return documents