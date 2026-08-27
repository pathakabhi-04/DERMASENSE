from src.rag.ingestion.schema import MedicalDocument
from src.rag.chunking.schema import DocumentChunk


def main():
    document = MedicalDocument(
        document_id="TEST_001",
        title="Test Medical Document",
        source="Test Source",
        source_url="https://example.com",
        text="This is a test medical document.",
        topic="test",
        condition="test_condition",
        sections=["test_section"],
    )

    chunk = DocumentChunk(
        chunk_id="TEST_001_000",
        document_id=document.document_id,
        text="This is a test chunk.",
        chunk_index=0,
        start_char=0,
        end_char=22,
        metadata={
            "source": document.source,
            "topic": document.topic,
        },
    )

    print("Document:")
    print(document)

    print("\nChunk:")
    print(chunk)


if __name__ == "__main__":
    main()