import unittest

from src.rag.retrieval.check_class_coverage import (
    find_class_coverage,
    load_documents,
)


class ClassCoverageTests(unittest.TestCase):
    def test_groups_chunks_by_document_id_not_shared_corpus_id(self):
        chunks = [
            {
                "document_id": "MEL_DOC",
                "text": "Melanoma information.",
                "metadata": {
                    "corpus_id": "dermasense-medical",
                    "title": "Melanoma source",
                },
            },
            {
                "document_id": "BCC_DOC",
                "text": "Basal cell carcinoma information.",
                "metadata": {
                    "corpus_id": "dermasense-medical",
                    "title": "BCC source",
                },
            },
        ]

        documents = load_documents(chunks)

        self.assertEqual(set(documents), {"MEL_DOC", "BCC_DOC"})
        self.assertEqual(documents["BCC_DOC"]["title"], "BCC source")

    def test_reports_covered_and_uncovered_native_classes(self):
        documents = load_documents(
            [
                {
                    "document_id": "LESION_DOC",
                    "text": (
                        "Actinic keratosis, basal cell carcinoma, melanoma, "
                        "a nevus, squamous cell carcinoma, and "
                        "seborrheic keratosis."
                    ),
                    "metadata": {"title": "Lesion source"},
                },
            ]
        )

        coverage = find_class_coverage(documents)

        self.assertTrue(all(coverage.values()))
        self.assertEqual(coverage["MEL"], ["LESION_DOC"])


if __name__ == "__main__":
    unittest.main()
