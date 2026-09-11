from __future__ import annotations

import json
from pathlib import Path

from src.rag.embeddings.embedder import SentenceTransformerEmbedder
from src.rag.retrieval.retriever import MedicalRetriever
from src.rag.vectorstore.faiss_store import FAISSVectorStore


CASES_PATH = Path("src/rag/retrieval/retrieval_cases.json")
INDEX_PATH = Path("data/rag/indexes/medical_v0.1")


def load_cases() -> list[dict]:
    with CASES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    print("=" * 80)
    print("DERMASENSE RAG - RETRIEVAL EVALUATION")
    print("=" * 80)

    print("\nLoading embedding model...")
    embedder = SentenceTransformerEmbedder()

    print("Loading persisted FAISS index...")
    store = FAISSVectorStore.load(INDEX_PATH)

    print(f"Index size: {store.size}")
    print(f"Embedding dimension: {embedder.dimension}")

    if store.size == 0:
        raise RuntimeError("Vector store is empty.")

    if store.index.d != embedder.dimension:
        raise RuntimeError(
            f"Dimension mismatch: index={store.index.d}, "
            f"embedder={embedder.dimension}"
        )

    retriever = MedicalRetriever(
        embedder=embedder,
        vector_store=store,
    )

    cases = load_cases()

    print(f"Evaluation cases: {len(cases)}")

    top1_hits = 0
    top3_hits = 0
    topic_top3_hits = 0

    failures = []

    for case_number, case in enumerate(cases, start=1):
        query = case["query"]
        expected_ids = set(case["expected_document_ids"])
        expected_topics = set(case["expected_topics"])

        results = retriever.search(
            query,
            top_k=3,
        )

        result_ids = [
            result.chunk.document_id
            for result in results
        ]

        result_topics = [
            result.chunk.metadata.get("topic")
            for result in results
        ]

        top1_hit = bool(
            results
            and results[0].chunk.document_id in expected_ids
        )

        top3_hit = any(
            document_id in expected_ids
            for document_id in result_ids
        )

        topic_top3_hit = any(
            topic in expected_topics
            for topic in result_topics
        )

        if top1_hit:
            top1_hits += 1

        if top3_hit:
            top3_hits += 1

        if topic_top3_hit:
            topic_top3_hits += 1

        print("\n" + "-" * 80)
        print(f"CASE {case_number}")
        print("-" * 80)
        print(f"Query: {query}")
        print(
            "Expected documents: "
            + ", ".join(sorted(expected_ids))
        )

        print("\nRetrieved:")

        for rank, result in enumerate(results, start=1):
            print(
                f"  {rank}. "
                f"score={result.score:.4f} "
                f"document={result.chunk.document_id} "
                f"topic={result.chunk.metadata.get('topic')}"
            )

        print(
            f"\nTop-1: {'PASS' if top1_hit else 'FAIL'}"
            f" | Top-3: {'PASS' if top3_hit else 'FAIL'}"
            f" | Topic Top-3: {'PASS' if topic_top3_hit else 'FAIL'}"
        )

        if not top3_hit:
            failures.append(
                {
                    "case": case_number,
                    "query": query,
                    "expected_document_ids": sorted(expected_ids),
                    "retrieved_document_ids": result_ids,
                    "retrieved_topics": result_topics,
                }
            )

    total = len(cases)

    top1_accuracy = top1_hits / total
    top3_accuracy = top3_hits / total
    topic_accuracy = topic_top3_hits / total

    print("\n" + "=" * 80)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 80)

    print(
        f"Top-1 document accuracy: "
        f"{top1_hits}/{total} "
        f"({top1_accuracy:.1%})"
    )

    print(
        f"Top-3 document accuracy: "
        f"{top3_hits}/{total} "
        f"({top3_accuracy:.1%})"
    )

    print(
        f"Top-3 topic accuracy: "
        f"{topic_top3_hits}/{total} "
        f"({topic_accuracy:.1%})"
    )

    if failures:
        print("\nFAILURES")
        print("=" * 80)

        for failure in failures:
            print(f"\nCase {failure['case']}: {failure['query']}")
            print(
                "Expected: "
                + ", ".join(failure["expected_document_ids"])
            )
            print(
                "Retrieved: "
                + ", ".join(failure["retrieved_document_ids"])
            )
    else:
        print("\nNo Top-3 document retrieval failures.")

    print("\nRetrieval evaluation completed.")


if __name__ == "__main__":
    main()
