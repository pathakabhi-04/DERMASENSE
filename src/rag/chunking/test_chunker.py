from src.rag.chunking.chunker import ParagraphChunker
from src.rag.ingestion.schema import MedicalDocument


def validate_offsets(
    document: MedicalDocument,
    chunks,
) -> None:
    """
    Verify that every chunk's character offsets point to the
    corresponding text in the original document.
    """

    for chunk in chunks:
        source_text = document.text[
            chunk.start_char:chunk.end_char
        ]

        normalized_source = "\n\n".join(
            part.strip()
            for part in source_text.split("\n\n")
            if part.strip()
        )

        assert normalized_source == chunk.text, (
            f"Offset validation failed for {chunk.chunk_id}"
        )


def print_chunks(
    title: str,
    document: MedicalDocument,
    chunks,
) -> None:
    print("\n" + "#" * 60)
    print(title)
    print("#" * 60)

    print(f"Document ID: {document.document_id}")
    print(f"Document length: {len(document.text)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "=" * 60)

        print(f"Chunk ID: {chunk.chunk_id}")
        print(
            f"Characters: "
            f"{chunk.start_char} -> {chunk.end_char}"
        )
        print(f"Length: {len(chunk.text)}")

        print(f"Metadata: {chunk.metadata}")

        print("\nText:")
        print(chunk.text)


def test_basic_chunking():
    """
    Basic unit test for paragraph-aware chunking.
    """

    text = (
        "Melanoma is a type of skin cancer.\n\n"
        "A changing mole may require professional evaluation.\n\n"
        "Changes can occur in size, shape, or color.\n\n"
        "This information should not be used to make a definitive diagnosis."
    )

    document = MedicalDocument(
        document_id="TEST_001",
        title="Test Medical Document",
        source="Test Source",
        source_url="https://example.com",
        text=text,
        topic="skin_cancer",
        condition="melanoma",
        sections=["warning_signs"],
        metadata={
            "corpus_id": "dermasense-medical",
            "corpus_version": "0.1",
        },
    )

    chunker = ParagraphChunker(
        max_chars=100,
        overlap_paragraphs=1,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) > 0

    # IDs should be deterministic.
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"TEST_001_{index:04d}"

    # Every chunk must belong to the original document.
    for chunk in chunks:
        assert chunk.document_id == document.document_id

    validate_offsets(document, chunks)

    print_chunks(
        "BASIC CHUNKING TEST",
        document,
        chunks,
    )

    print("\nBasic chunking test passed.")


def test_medical_chunking():
    """
    Test chunking using a more realistic medical document.
    """

    medical_text = (
        "Melanoma warning signs\n\n"

        "A changing mole may be a warning sign. "
        "Changes can include changes in size, shape, or color.\n\n"

        "The ABCDE method can help people recognize "
        "features of a mole that may warrant professional "
        "medical evaluation.\n\n"

        "A: Asymmetry means one half of the mole does not "
        "match the other half.\n\n"

        "B: Border refers to the edge of the mole. "
        "An irregular or poorly defined border can be concerning.\n\n"

        "C: Color refers to the color of the mole. "
        "Multiple colors or an uneven distribution of color "
        "may be a warning sign.\n\n"

        "D: Diameter refers to the size of the mole.\n\n"

        "E: Evolving refers to changes in the mole over time.\n\n"

        "This information should not be used to make a "
        "definitive diagnosis."
    )

    document = MedicalDocument(
        document_id="TEST_MEDICAL_001",
        title="Test Melanoma Warning Signs",
        source="Test Medical Source",
        source_url="https://example.com/test",
        text=medical_text,
        topic="skin_cancer",
        condition="melanoma",
        sections=[
            "warning_signs",
            "ABCDE",
        ],
        metadata={
            "corpus_id": "dermasense-medical",
            "corpus_version": "0.1",
        },
    )

    chunker = ParagraphChunker(
        max_chars=300,
        overlap_paragraphs=1,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) > 0

    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == (
            f"TEST_MEDICAL_001_{index:04d}"
        )

        assert chunk.document_id == document.document_id

        assert chunk.metadata["topic"] == "skin_cancer"
        assert chunk.metadata["condition"] == "melanoma"

    validate_offsets(document, chunks)

    print_chunks(
        "MEDICAL CHUNKING TEST",
        document,
        chunks,
    )

    print("\nMedical chunking test passed.")


def test_overlap():
    """
    Verify that adjacent chunks share contextual content
    when overlap is enabled.
    """

    text = (
        "Paragraph one contains information.\n\n"
        "Paragraph two contains important context.\n\n"
        "Paragraph three contains additional information.\n\n"
        "Paragraph four contains the final information."
    )

    document = MedicalDocument(
        document_id="TEST_OVERLAP_001",
        title="Overlap Test",
        source="Test Source",
        source_url=None,
        text=text,
        topic="test",
        condition=None,
        sections=[],
    )

    chunker = ParagraphChunker(
        max_chars=70,
        overlap_paragraphs=1,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2

    # Adjacent chunks should share text.
    for previous, current in zip(
        chunks,
        chunks[1:],
    ):
        previous_paragraphs = previous.text.split("\n\n")
        current_paragraphs = current.text.split("\n\n")

        assert (
            previous_paragraphs[-1]
            == current_paragraphs[0]
        )

    print("\nOverlap test passed.")


def main():
    test_basic_chunking()
    test_medical_chunking()
    test_overlap()

    print("\n" + "=" * 60)
    print("ALL CHUNKING TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()