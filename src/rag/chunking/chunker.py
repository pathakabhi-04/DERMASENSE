from src.rag.chunking.schema import DocumentChunk
from src.rag.ingestion.schema import MedicalDocument


class ParagraphChunker:
    """
    Splits a MedicalDocument into paragraph-aware chunks.

    Chunks are built from complete paragraphs where possible.
    Adjacent chunks may overlap by a configurable number of paragraphs
    to preserve local context during retrieval.
    """

    def __init__(
        self,
        max_chars: int = 1200,
        overlap_paragraphs: int = 1,
    ):
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")

        if overlap_paragraphs < 0:
            raise ValueError("overlap_paragraphs cannot be negative")

        self.max_chars = max_chars
        self.overlap_paragraphs = overlap_paragraphs

    def chunk(self, document: MedicalDocument) -> list[DocumentChunk]:
        """
        Convert a MedicalDocument into retrieval-ready chunks.
        """

        paragraphs = self._split_paragraphs(document.text)

        if not paragraphs:
            return []

        chunks: list[DocumentChunk] = []

        current: list[tuple[int, int, str]] = []
        current_length = 0

        for start, end, paragraph in paragraphs:

            paragraph_length = len(paragraph)

            separator_length = 2 if current else 0

            would_exceed = (
                current
                and current_length
                + separator_length
                + paragraph_length
                > self.max_chars
            )

            if would_exceed:
                chunks.append(
                    self._build_chunk(
                        document=document,
                        paragraphs=current,
                        chunk_index=len(chunks),
                    )
                )

                if self.overlap_paragraphs > 0:
                    current = current[-self.overlap_paragraphs :]

                    current_length = self._joined_length(current)
                else:
                    current = []
                    current_length = 0

            separator_length = 2 if current else 0

            current.append((start, end, paragraph))
            current_length += separator_length + paragraph_length

        if current:
            chunks.append(
                self._build_chunk(
                    document=document,
                    paragraphs=current,
                    chunk_index=len(chunks),
                )
            )

        return chunks

    @staticmethod
    def _split_paragraphs(
        text: str,
    ) -> list[tuple[int, int, str]]:
        """
        Split text into paragraphs while preserving exact character offsets.

        Returns:
            (start_offset, end_offset, paragraph_text)
        """

        paragraphs = []

        cursor = 0

        for block in text.split("\n\n"):
            block_start = cursor
            block_end = cursor + len(block)

            stripped = block.strip()

            if stripped:
                leading_whitespace = len(block) - len(block.lstrip())
                trailing_whitespace = len(block) - len(block.rstrip())

                start = block_start + leading_whitespace
                end = block_end - trailing_whitespace

                paragraphs.append(
                    (
                        start,
                        end,
                        text[start:end],
                    )
                )

            cursor = block_end + 2

        return paragraphs

    @staticmethod
    def _joined_length(
        paragraphs: list[tuple[int, int, str]],
    ) -> int:
        """
        Calculate the character length of paragraphs when joined by
        two newline characters.
        """

        if not paragraphs:
            return 0

        return sum(
            len(paragraph)
            for _, _, paragraph in paragraphs
        ) + (len(paragraphs) - 1) * 2

    @staticmethod
    def _build_chunk(
        document: MedicalDocument,
        paragraphs: list[tuple[int, int, str]],
        chunk_index: int,
    ) -> DocumentChunk:

        text = "\n\n".join(
            paragraph
            for _, _, paragraph in paragraphs
        )

        start_char = paragraphs[0][0]
        end_char = paragraphs[-1][1]

        metadata = {
            **document.metadata,
            "source": document.source,
            "source_url": document.source_url,
            "title": document.title,
            "topic": document.topic,
            "condition": document.condition,
            "sections": document.sections,
        }

        return DocumentChunk(
            chunk_id=f"{document.document_id}_{chunk_index:04d}",
            document_id=document.document_id,
            text=text,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata,
        )