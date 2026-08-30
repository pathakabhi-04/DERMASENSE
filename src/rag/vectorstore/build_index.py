from __future__ import annotations

import json
from pathlib import Path

from src.rag.chunking.corpus_chunker import MedicalCorpusChunker
from src.rag.embeddings.embedder import SentenceTransformerEmbedder
from src.rag.vectorstore.faiss_store import FAISSVectorStore


def main():

    print("=" * 70)
    print("DERMASENSE RAG — BUILDING MEDICAL INDEX")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Chunk the acquired medical corpus
    # ---------------------------------------------------------

    print("\n[1/4] Loading and chunking corpus...")

    chunk_pipeline = MedicalCorpusChunker(
        manifest_path="data/rag/corpus_manifest.json",
        acquisition_manifest_path=(
            "data/rag/acquisition_manifest.json"
        ),
        raw_dir="data/rag/raw",
        max_chars=1200,
        overlap_paragraphs=1,
    )

    chunks = chunk_pipeline.chunk_corpus()

    print(
        f"Chunks created: {len(chunks)}"
    )

    # ---------------------------------------------------------
    # 2. Create embeddings
    # ---------------------------------------------------------

    print("\n[2/4] Loading embedding model...")

    embedder = SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(
        f"Embedding dimension: "
        f"{embedder.dimension}"
    )

    texts = [
        chunk.text
        for chunk in chunks
    ]

    print(
        f"Embedding {len(texts)} chunks..."
    )

    embeddings = embedder.embed_documents(
        texts
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # ---------------------------------------------------------
    # 3. Build FAISS index
    # ---------------------------------------------------------

    print("\n[3/4] Building FAISS index...")

    store = FAISSVectorStore(
        dimension=embedder.dimension
    )

    store.add(
        embeddings,
        chunks,
    )

    print(
        f"Vector store size: "
        f"{store.size}"
    )

    # ---------------------------------------------------------
    # 4. Persist index + metadata
    # ---------------------------------------------------------

    print("\n[4/4] Persisting index...")

    index_dir = Path(
        "data/rag/indexes/medical_v0.1"
    )

    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    store.save(
        index_dir
    )

    # Save build metadata separately.
    build_metadata = {
        "corpus_id": "dermasense-medical",
        "corpus_version": "0.1",
        "embedding_model": (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        "embedding_dimension": embedder.dimension,
        "chunk_count": len(chunks),
        "chunk_max_chars": 1200,
        "chunk_overlap_paragraphs": 1,
    }

    metadata_path = (
        index_dir / "build_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            build_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Index saved to: {index_dir}"
    )

    print(
        f"Metadata saved to: {metadata_path}"
    )

    print("\n" + "=" * 70)
    print("RAG INDEX BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()