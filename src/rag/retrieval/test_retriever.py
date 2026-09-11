from src.rag.chunking.schema import DocumentChunk
from src.rag.embeddings.embedder import SentenceTransformerEmbedder
from src.rag.retrieval.retriever import MedicalRetriever
from src.rag.vectorstore.faiss_store import FAISSVectorStore


def make_chunk(
    chunk_id: str,
    text: str,
    topic: str,
    condition: str | None = None,
) -> DocumentChunk:

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="TEST_DOC",
        text=text,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        metadata={
            "source": "Test Medical Source",
            "source_url": "https://example.com",
            "title": "Test Medical Document",
            "topic": topic,
            "condition": condition,
        },
    )


def main():
    print("Loading embedding model...")

    embedder = SentenceTransformerEmbedder()

    chunks = [
        make_chunk(
            "TEST_0000",
            (
                "A changing mole may require professional "
                "evaluation. Changes can occur in size, "
                "shape, or color."
            ),
            topic="skin_lesions",
            condition="melanoma",
        ),
        make_chunk(
            "TEST_0001",
            (
                "A superficial scrape can be cleaned and "
                "protected with a dressing."
            ),
            topic="wounds",
            condition="abrasion",
        ),
        make_chunk(
            "TEST_0002",
            (
                "Burn first aid depends on the severity "
                "and extent of the burn."
            ),
            topic="burns",
            condition="burn",
        ),
    ]

    print(
        f"Embedding {len(chunks)} test chunks..."
    )

    embeddings = embedder.embed_documents(
        [chunk.text for chunk in chunks]
    )

    store = FAISSVectorStore(
        dimension=embedder.dimension
    )

    store.add(
        embeddings,
        chunks,
    )

    print(
        f"Vector store size: {store.size}"
    )

    retriever = MedicalRetriever(
        embedder=embedder,
        vector_store=store,
    )

    queries = [
        "What changes in a mole should I pay attention to?",
        "What should I do for a superficial scrape?",
        "What should I know about burn first aid?",
    ]

    for query in queries:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = retriever.search(
            query,
            top_k=3,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"\n{rank}. "
                f"score={result.score:.4f}"
            )

            print(
                f"   chunk={result.chunk.chunk_id}"
            )

            print(
                f"   topic={result.chunk.metadata['topic']}"
            )

            print(
                f"   condition="
                f"{result.chunk.metadata['condition']}"
            )

            print(
                f"   text={result.chunk.text}"
            )

    print("\nRetriever test completed successfully.")


if __name__ == "__main__":
    main()