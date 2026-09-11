from __future__ import annotations

from dataclasses import dataclass

from src.rag.retrieval.retriever import MedicalRetriever, RetrievalResult

DEFAULT_MAX_CHUNKS = 3
DEFAULT_CANDIDATE_POOL = 10


@dataclass
class EvidenceChunk:
    """
    One piece of retrieved evidence selected for the prompt.
    """

    score: float
    document_id: str
    title: str
    text: str


@dataclass
class EvidenceBundle:
    """
    Evidence assembled for one query: the selected chunks, plus
    rendering helpers for the prompt builder and the baseline
    citation line.
    """

    chunks: list[EvidenceChunk]

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    @property
    def top_score(self) -> float | None:
        """
        Highest similarity score among the selected chunks, or None
        if nothing was retrieved. Downstream components (prompt
        builder, safety layer) use this to decide when to state
        uncertainty explicitly — this formatter does not apply that
        threshold itself.
        """

        if self.is_empty:
            return None

        return max(chunk.score for chunk in self.chunks)

    def format_for_prompt(self) -> str:
        """
        Render evidence as numbered, source-attributed blocks
        suitable for embedding in the LLM prompt.
        """

        if self.is_empty:
            return "No relevant evidence was retrieved."

        blocks = [
            f"[{index}] Source: {chunk.title}\n{chunk.text}"
            for index, chunk in enumerate(self.chunks, start=1)
        ]

        return "\n\n".join(blocks)

    def format_sources_line(self) -> str:
        """
        Baseline citation format (spec section 6):
        "Sources: [document titles]", deduplicated, order preserved.
        """

        if self.is_empty:
            return "Sources: none"

        titles: list[str] = []

        for chunk in self.chunks:
            if chunk.title not in titles:
                titles.append(chunk.title)

        return "Sources: " + "; ".join(titles)


def select_top_evidence(
    results: list[RetrievalResult],
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> list[EvidenceChunk]:
    """
    Select up to max_chunks chunks from ranked retrieval results,
    deduplicated by source document.

    This implements the evidence-selection policy in the primary
    specification (section 3.1): feed the top-3 chunks, deduplicated
    by source document, so that a single document's chunks cannot
    crowd out a second relevant document.

    Assumes `results` is already ranked by descending relevance.
    """

    if max_chunks <= 0:
        raise ValueError(
            "max_chunks must be greater than zero."
        )

    selected: list[EvidenceChunk] = []
    seen_documents: set[str] = set()

    for result in results:
        document_id = result.chunk.document_id

        if document_id in seen_documents:
            continue

        selected.append(
            EvidenceChunk(
                score=result.score,
                document_id=document_id,
                title=result.chunk.metadata.get(
                    "title",
                    document_id,
                ),
                text=result.chunk.text,
            )
        )

        seen_documents.add(document_id)

        if len(selected) >= max_chunks:
            break

    return selected


class EvidenceFormatter:
    """
    Retrieves and formats medical evidence for the RAG prompt
    builder, applying the top-3-deduplicated-by-document policy.
    """

    def __init__(
        self,
        retriever: MedicalRetriever,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    ):
        if candidate_pool < max_chunks:
            raise ValueError(
                "candidate_pool must be at least max_chunks, "
                "or deduplication has nothing extra to draw from."
            )

        self.retriever = retriever
        self.max_chunks = max_chunks
        self.candidate_pool = candidate_pool

    def get_evidence(self, query: str) -> EvidenceBundle:
        """
        Retrieve a wider candidate pool, then select the top-3
        chunks deduplicated by document.
        """

        results = self.retriever.search(
            query,
            top_k=self.candidate_pool,
        )

        selected = select_top_evidence(
            results,
            max_chunks=self.max_chunks,
        )

        return EvidenceBundle(chunks=selected)
