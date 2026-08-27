import numpy as np

from src.rag.embeddings.embedder import (
    SentenceTransformerEmbedder,
)


def main():
    print("Loading embedding model...")

    embedder = SentenceTransformerEmbedder()

    print(f"Model: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.dimension}")

    documents = [
        "A changing mole may require professional evaluation.",
        "A superficial scrape can be cleaned and protected with a dressing.",
        "Burns may require first aid and medical evaluation depending on severity.",
    ]

    print("\nEmbedding documents...")

    document_embeddings = embedder.embed_documents(
        documents
    )

    print(
        f"Document embedding shape: "
        f"{document_embeddings.shape}"
    )

    print(
        f"Embedding dtype: "
        f"{document_embeddings.dtype}"
    )

    print(
        f"Embedding norm: "
        f"{np.linalg.norm(document_embeddings, axis=1)}"
    )

    query = (
        "What should I know about a changing skin lesion?"
    )

    query_embedding = embedder.embed_query(query)

    print(
        f"\nQuery embedding shape: "
        f"{query_embedding.shape}"
    )

    print(
        f"Query embedding norm: "
        f"{np.linalg.norm(query_embedding):.6f}"
    )

    similarities = (
        document_embeddings @ query_embedding
    )

    print("\nSimilarity scores:")

    for document, score in zip(
        documents,
        similarities,
    ):
        print(
            f"{score:.4f}  |  {document}"
        )


if __name__ == "__main__":
    main()