from __future__ import annotations

from pathlib import Path

from src.rag.chunking.chunker import ParagraphChunker
from src.rag.chunking.schema import DocumentChunk
from src.rag.ingestion.corpus_loader import MedicalCorpusLoader


class CorpusChunkingError(RuntimeError):
    """Raised when the medical corpus cannot be chunked safely."""


class MedicalCorpusChunker:
    """
    Orchestrates chunking across the complete DermaSense medical corpus.

    The underlying ParagraphChunker remains responsible for the actual
    paragraph-aware splitting logic.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        acquisition_manifest_path: str | Path,
        raw_dir: str | Path,
        max_chars: int = 1200,
        overlap_paragraphs: int = 1,
    ):
        self.loader = MedicalCorpusLoader(
            manifest_path=manifest_path,
            acquisition_manifest_path=acquisition_manifest_path,
            raw_dir=raw_dir,
        )

        self.chunker = ParagraphChunker(
            max_chars=max_chars,
            overlap_paragraphs=overlap_paragraphs,
        )

    def chunk_corpus(self) -> list[DocumentChunk]:
        """
        Load the acquired corpus and chunk every document.
        """

        documents = self.loader.load()

        if not documents:
            raise CorpusChunkingError(
                "No medical documents were loaded."
            )

        all_chunks: list[DocumentChunk] = []

        for document in documents:
            chunks = self.chunker.chunk(
                document
            )

            if not chunks:
                raise CorpusChunkingError(
                    f"No chunks generated for "
                    f"{document.document_id}"
                )

            all_chunks.extend(
                chunks
            )

        return all_chunks