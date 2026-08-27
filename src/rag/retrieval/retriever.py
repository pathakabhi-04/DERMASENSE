from __future__ import annotations

from dataclasses import dataclass

from src.rag.chunking.schema import DocumentChunk
from src.rag.embeddings.embedder import SentenceTransformerEmbedder
from src.rag.vectorstore.faiss_store import FAISSVectorStore


@dataclass
class RetrievalResult:
    """
    A single retrieved document chunk and its similarity score.
    """

    score: float
    chunk: DocumentChunk


class MedicalRetriever:
    """
    Semantic retriever for the DermaSense medical corpus.

    Query flow:

        user query
            ↓
        embedding
            ↓
        FAISS search
            ↓
        ranked DocumentChunks
    """

    def __init__(
        self,
        embedder: SentenceTransformerEmbedder,
        vector_store: FAISSVectorStore,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve the top-k semantically relevant chunks.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_embedding = self.embedder.embed_query(
            query
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        return [
            RetrievalResult(
                score=score,
                chunk=chunk,
            )
            for score, chunk in results
        ]