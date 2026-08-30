from src.rag.ingestion.corpus_loader import (
    MedicalCorpusLoader,
)


def main():

    loader = MedicalCorpusLoader(
        manifest_path="data/rag/corpus_manifest.json",
        acquisition_manifest_path=(
            "data/rag/acquisition_manifest.json"
        ),
        raw_dir="data/rag/raw",
    )

    documents = loader.load()

    print(
        f"Documents loaded: {len(documents)}"
    )

    assert len(documents) == 8, (
        "Expected 8 successfully acquired documents."
    )

    for document in documents:

        print("\n" + "=" * 70)

        print(
            f"ID: {document.document_id}"
        )

        print(
            f"Title: {document.title}"
        )

        print(
            f"Source: {document.source}"
        )

        print(
            f"Topic: {document.topic}"
        )

        print(
            f"Condition: {document.condition}"
        )

        print(
            f"Sections: {document.sections}"
        )

        print(
            f"Text characters: {len(document.text)}"
        )

        print(
            f"SHA-256: "
            f"{document.metadata['sha256']}"
        )

        print("\nText preview:")

        print(
            document.text[:500]
        )

    document_ids = {
        document.document_id
        for document in documents
    }

    expected_ids = {
        "AAD_MOLES_SYMPTOMS_001",
        "NCI_MOLES_MELANOMA_001",
        "NCI_MELANOMA_APPEARANCE_001",
        "AAD_MOLE_PROBLEM_001",
        "MEDLINEPLUS_SCRAPE_001",
        "MEDLINEPLUS_CUTS_001",
        "MEDLINEPLUS_BURNS_001",
        "MEDLINEPLUS_MINOR_BURNS_001",
    }

    assert document_ids == expected_ids

    print(
        "\n" + "#" * 70
    )

    print(
        "CORPUS LOADER TEST PASSED"
    )

    print(
        "#" * 70
    )


if __name__ == "__main__":
    main()