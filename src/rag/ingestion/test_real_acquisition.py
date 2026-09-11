from pathlib import Path

from src.rag.ingestion.acquisition import (
    download_source,
    load_manifest,
)


def main():
    manifest_path = Path(
        "data/rag/corpus_manifest.json"
    )

    output_dir = Path(
        "data/rag/raw"
    )

    manifest = load_manifest(
        manifest_path
    )

    document = manifest["documents"][0]

    print("Acquiring:")
    print(f"  ID: {document['document_id']}")
    print(f"  Title: {document['title']}")
    print(f"  Source: {document['source']}")
    print(f"  URL: {document['source_url']}")

    result = download_source(
        document=document,
        output_dir=output_dir,
    )

    print("\nAcquisition successful:")
    print(f"  Status: {result['http_status']}")
    print(f"  Content type: {result['content_type']}")
    print(f"  Bytes: {result['byte_size']}")
    print(f"  SHA-256: {result['sha256']}")
    print(f"  Saved to: {result['local_path']}")


if __name__ == "__main__":
    main()