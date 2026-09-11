from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.rag.chunking.schema import DocumentChunk


class FAISSVectorStore:
    """
    FAISS-backed vector store for DermaSense document chunks.

    Vectors are expected to be L2-normalized so that
    inner product corresponds to cosine similarity.
    """

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError(
                "Embedding dimension must be greater than zero."
            )

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self.chunks: list[DocumentChunk] = []

    def add(
        self,
        embeddings: np.ndarray,
        chunks: list[DocumentChunk],
    ) -> None:
        """
        Add embeddings and their corresponding chunks.

        The FAISS vector position and chunk list position must remain
        aligned.
        """

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks."
            )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must have shape (n, dimension)."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {embeddings.shape[1]}."
            )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self.index.add(embeddings)

        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[float, DocumentChunk]]:
        """
        Search the vector store.

        Returns:
            List of (similarity_score, DocumentChunk)
        """

        if self.index.ntotal == 0:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape != (
            1,
            self.dimension,
        ):
            raise ValueError(
                f"Expected query shape "
                f"(1, {self.dimension}), "
                f"got {query_embedding.shape}."
            )

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index < 0:
                continue

            results.append(
                (
                    float(score),
                    self.chunks[index],
                )
            )

        return results

    def save(self, directory: str | Path) -> None:
        """
        Persist FAISS index and chunk metadata.
        """

        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_path = directory / "medical.faiss"
        metadata_path = directory / "chunks.json"

        faiss.write_index(
            self.index,
            str(index_path),
        )

        serialized_chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": chunk.metadata,
            }
            for chunk in self.chunks
        ]

        metadata_path.write_text(
            json.dumps(
                serialized_chunks,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "FAISSVectorStore":
        """
        Load a previously persisted vector store.
        """

        directory = Path(directory)

        index_path = directory / "medical.faiss"
        metadata_path = directory / "chunks.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found: {metadata_path}"
            )

        index = faiss.read_index(
            str(index_path)
        )

        raw_chunks = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        store = cls(
            dimension=index.d
        )

        store.index = index

        store.chunks = [
            DocumentChunk(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                text=item["text"],
                chunk_index=item["chunk_index"],
                start_char=item["start_char"],
                end_char=item["end_char"],
                metadata=item["metadata"],
            )
            for item in raw_chunks
        ]

        if store.index.ntotal != len(store.chunks):
            raise ValueError(
                "FAISS index size does not match "
                "stored chunk metadata."
            )

        return store

    @property
    def size(self) -> int:
        return self.index.ntotal