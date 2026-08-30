from pathlib import Path
import tempfile

from src.rag.ingestion.acquisition import (
    download_source,
    load_manifest,
)


def main():
    manifest_path = Path(
        "data/rag/corpus_manifest.json"
    )

    manifest = load_manifest(
        manifest_path
    )

    print(
        f"Corpus: {manifest['corpus_id']}"
    )

    print(
        f"Version: {manifest['version']}"
    )

    print(
        f"Documents: {len(manifest['documents'])}"
    )

    assert len(manifest["documents"]) == 10

    first_document = manifest["documents"][0]

    print("\nTesting first manifest entry:")
    print(
        f"ID: {first_document['document_id']}"
    )
    print(
        f"Title: {first_document['title']}"
    )
    print(
        f"Source: {first_document['source']}"
    )
    print(
        f"URL: {first_document['source_url']}"
    )

    # We are only testing manifest parsing here.
    # Actual network acquisition will be done separately.
    assert first_document["document_id"]
    assert first_document["source_url"]

    print(
        "\nManifest acquisition test passed."
    )


if __name__ == "__main__":
    main()