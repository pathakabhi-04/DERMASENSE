from pathlib import Path

from src.rag.ingestion.acquisition import (
    download_source,
    load_manifest,
)
from src.rag.ingestion.html_extractor import (
    MedicalHTMLExtractor,
)


def main():
    manifest_path = Path(
        "data/rag/corpus_manifest.json"
    )

    raw_dir = Path(
        "data/rag/raw"
    )

    manifest = load_manifest(
        manifest_path
    )

    documents = manifest["documents"]

    # One representative document from each source family.
    source_families = {
        "AAD": "AAD_MOLES_SYMPTOMS_001",
        "NCI": "NCI_MOLES_MELANOMA_001",
        "MedlinePlus": "MEDLINEPLUS_SCRAPE_001",
        "NHS": "NHS_FIRST_AID_001",
    }

    documents_by_id = {
        document["document_id"]: document
        for document in documents
    }

    extractor = MedicalHTMLExtractor()

    for family, document_id in source_families.items():

        print("\n" + "#" * 70)
        print(f"SOURCE FAMILY: {family}")
        print("#" * 70)

        document = documents_by_id[document_id]

        result = download_source(
            document=document,
            output_dir=raw_dir,
        )

        print(
            f"Downloaded: {result['local_path']}"
        )

        html = Path(
            result["local_path"]
        ).read_text(
            encoding="utf-8"
        )

        text = extractor.extract(
            html
        )

        print(
            f"Extracted characters: {len(text)}"
        )

        print("\nFirst 3000 characters:")
        print("-" * 70)
        print(text[:3000])

        assert len(text) > 0

        print(
            f"\n{family} extraction passed."
        )

    print(
        "\n" + "=" * 70
    )
    print(
        "ALL SOURCE FAMILY TESTS PASSED"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()