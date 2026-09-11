import tempfile
from pathlib import Path

import numpy as np

from src.rag.chunking.schema import DocumentChunk
from src.rag.vectorstore.faiss_store import FAISSVectorStore


def create_chunk(
    chunk_id: str,
    text: str,
) -> DocumentChunk:

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="TEST_DOC",
        text=text,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        metadata={
            "source": "Test Source",
            "topic": "test",
        },
    )


def main():
    # Three simple normalized vectors.
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    chunks = [
        create_chunk(
            "TEST_0000",
            "Information about melanoma.",
        ),
        create_chunk(
            "TEST_0001",
            "Information about scratches.",
        ),
        create_chunk(
            "TEST_0002",
            "Information about burns.",
        ),
    ]

    store = FAISSVectorStore(
        dimension=3
    )

    store.add(
        embeddings,
        chunks,
    )

    print(f"Store size: {store.size}")

    # Query should be closest to the first vector.
    query = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    results = store.search(
        query,
        top_k=3,
    )

    print("\nSearch results:")

    for score, chunk in results:
        print(
            f"{score:.4f} | "
            f"{chunk.chunk_id} | "
            f"{chunk.text}"
        )

    assert results[0][1].chunk_id == "TEST_0000"

    print("\nSearch test passed.")

    # Test persistence.
    with tempfile.TemporaryDirectory() as temp_dir:

        store.save(temp_dir)

        loaded = FAISSVectorStore.load(
            temp_dir
        )

        print(
            f"\nLoaded store size: "
            f"{loaded.size}"
        )

        loaded_results = loaded.search(
            query,
            top_k=1,
        )

        assert (
            loaded_results[0][1].chunk_id
            == "TEST_0000"
        )

        print(
            "Persistence test passed."
        )


if __name__ == "__main__":
    main()