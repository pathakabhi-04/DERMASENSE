from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """
    Embedding wrapper for the DermaSense RAG pipeline.

    The wrapper keeps the rest of the RAG system independent
    from the specific sentence-transformers model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""

        return self.model.get_embedding_dimension()

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Embed multiple document chunks.

        Returns:
            float32 numpy array with shape:
            (number_of_texts, embedding_dimension)
        """

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single user query.

        Returns:
            float32 numpy array with shape:
            (embedding_dimension,)
        """

        if not text.strip():
            raise ValueError(
                "Query text cannot be empty."
            )

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )