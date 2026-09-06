import unittest

from src.rag.chunking.schema import DocumentChunk
from src.rag.retrieval.evidence import (
    EvidenceBundle,
    select_top_evidence,
)
from src.rag.retrieval.retriever import RetrievalResult


def make_result(
    document_id: str,
    chunk_id: str,
    score: float,
    title: str | None = None,
    text: str = "Evidence text.",
) -> RetrievalResult:

    return RetrievalResult(
        score=score,
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=text,
            chunk_index=0,
            start_char=0,
            end_char=len(text),
            metadata={"title": title or document_id},
        ),
    )


class SelectTopEvidenceTests(unittest.TestCase):
    def test_deduplicates_by_document_id(self):
        results = [
            make_result("DOC_A", "DOC_A_0", 0.9),
            make_result("DOC_A", "DOC_A_1", 0.85),
            make_result("DOC_B", "DOC_B_0", 0.8),
        ]

        selected = select_top_evidence(results, max_chunks=3)

        self.assertEqual(
            [chunk.document_id for chunk in selected],
            ["DOC_A", "DOC_B"],
        )

    def test_does_not_let_one_document_crowd_out_a_second(self):
        results = [
            make_result("DOC_A", "DOC_A_0", 0.95),
            make_result("DOC_A", "DOC_A_1", 0.94),
            make_result("DOC_A", "DOC_A_2", 0.93),
            make_result("DOC_B", "DOC_B_0", 0.50),
        ]

        selected = select_top_evidence(results, max_chunks=3)

        document_ids = [chunk.document_id for chunk in selected]

        self.assertIn("DOC_B", document_ids)
        self.assertEqual(len(selected), 2)

    def test_respects_max_chunks_across_distinct_documents(self):
        results = [
            make_result("DOC_A", "DOC_A_0", 0.9),
            make_result("DOC_B", "DOC_B_0", 0.8),
            make_result("DOC_C", "DOC_C_0", 0.7),
            make_result("DOC_D", "DOC_D_0", 0.6),
        ]

        selected = select_top_evidence(results, max_chunks=3)

        self.assertEqual(len(selected), 3)
        self.assertEqual(
            [chunk.document_id for chunk in selected],
            ["DOC_A", "DOC_B", "DOC_C"],
        )

    def test_empty_results_produce_empty_selection(self):
        self.assertEqual(select_top_evidence([], max_chunks=3), [])

    def test_rejects_non_positive_max_chunks(self):
        with self.assertRaises(ValueError):
            select_top_evidence([], max_chunks=0)


class EvidenceBundleTests(unittest.TestCase):
    def test_empty_bundle_formatting(self):
        bundle = EvidenceBundle(chunks=[])

        self.assertTrue(bundle.is_empty)
        self.assertIsNone(bundle.top_score)
        self.assertEqual(
            bundle.format_for_prompt(),
            "No relevant evidence was retrieved.",
        )
        self.assertEqual(
            bundle.format_sources_line(),
            "Sources: none",
        )

    def test_format_for_prompt_includes_source_and_text(self):
        results = [
            make_result(
                "DOC_A",
                "DOC_A_0",
                0.9,
                title="Melanoma: Signs and symptoms",
                text="Watch for asymmetry, border irregularity.",
            ),
        ]

        bundle = EvidenceBundle(
            chunks=select_top_evidence(results, max_chunks=3)
        )

        rendered = bundle.format_for_prompt()

        self.assertIn("Melanoma: Signs and symptoms", rendered)
        self.assertIn(
            "Watch for asymmetry, border irregularity.",
            rendered,
        )
        self.assertTrue(rendered.startswith("[1]"))

    def test_format_sources_line_deduplicates_titles(self):
        results = [
            make_result("DOC_A", "DOC_A_0", 0.9, title="Shared Title"),
            make_result("DOC_B", "DOC_B_0", 0.8, title="Shared Title"),
            make_result("DOC_C", "DOC_C_0", 0.7, title="Other Title"),
        ]

        bundle = EvidenceBundle(
            chunks=select_top_evidence(results, max_chunks=3)
        )

        self.assertEqual(
            bundle.format_sources_line(),
            "Sources: Shared Title; Other Title",
        )

    def test_top_score_is_the_max_selected_score(self):
        results = [
            make_result("DOC_A", "DOC_A_0", 0.7),
            make_result("DOC_B", "DOC_B_0", 0.9),
        ]

        bundle = EvidenceBundle(
            chunks=select_top_evidence(results, max_chunks=3)
        )

        self.assertEqual(bundle.top_score, 0.9)


if __name__ == "__main__":
    unittest.main()
