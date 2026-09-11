from collections import Counter

from src.rag.chunking.corpus_chunker import (
    MedicalCorpusChunker,
)


def main():

    pipeline = MedicalCorpusChunker(
        manifest_path=(
            "data/rag/corpus_manifest.json"
        ),
        acquisition_manifest_path=(
            "data/rag/acquisition_manifest.json"
        ),
        raw_dir="data/rag/raw",
        max_chars=1200,
        overlap_paragraphs=1,
    )

    chunks = pipeline.chunk_corpus()

    print(
        f"Total chunks: {len(chunks)}"
    )

    assert len(chunks) > 0

    # ---------------------------------------------------------
    # Document coverage
    # ---------------------------------------------------------

    document_ids = {
        chunk.document_id
        for chunk in chunks
    }

    expected_document_ids = {
        "AAD_MOLES_SYMPTOMS_001",
        "NCI_MOLES_MELANOMA_001",
        "NCI_MELANOMA_APPEARANCE_001",
        "AAD_MOLE_PROBLEM_001",
        "MEDLINEPLUS_SCRAPE_001",
        "MEDLINEPLUS_CUTS_001",
        "MEDLINEPLUS_BURNS_001",
        "MEDLINEPLUS_MINOR_BURNS_001",
    }

    assert document_ids == expected_document_ids

    # ---------------------------------------------------------
    # Chunk ID uniqueness
    # ---------------------------------------------------------

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(
        set(chunk_ids)
    )

    # ---------------------------------------------------------
    # Metadata validation
    # ---------------------------------------------------------

    for chunk in chunks:

        assert chunk.document_id
        assert chunk.text.strip()

        assert chunk.metadata[
            "corpus_id"
        ] == "dermasense-medical"

        assert chunk.metadata[
            "corpus_version"
        ] == "0.1"

        assert chunk.metadata[
            "source"
        ]

        assert chunk.metadata[
            "source_url"
        ]

        assert chunk.metadata[
            "title"
        ]

        assert chunk.metadata[
            "topic"
        ]

        assert chunk.metadata[
            "sha256"
        ]

    # ---------------------------------------------------------
    # Print corpus statistics
    # ---------------------------------------------------------

    counts = Counter(
        chunk.document_id
        for chunk in chunks
    )

    print(
        "\nChunks per document:"
    )

    for document_id, count in counts.items():
        print(
            f"  {document_id}: {count}"
        )

    print(
        "\nSample chunks:"
    )

    for chunk in chunks[:5]:

        print(
            "\n" + "=" * 70
        )

        print(
            f"ID: {chunk.chunk_id}"
        )

        print(
            f"Length: {len(chunk.text)}"
        )

        print(
            f"Topic: {chunk.metadata['topic']}"
        )

        print(
            f"Condition: {chunk.metadata['condition']}"
        )

        print(
            f"Source: {chunk.metadata['source']}"
        )

        print(
            f"\n{chunk.text[:1000]}"
        )

    print(
        "\n" + "#" * 70
    )

    print(
        "CORPUS CHUNKING TEST PASSED"
    )

    print(
        "#" * 70
    )


if __name__ == "__main__":
    main()