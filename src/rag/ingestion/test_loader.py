from pathlib import Path

from src.rag.ingestion.loader import CorpusManifestLoader


def main():
    manifest_path = Path("data/rag/corpus_manifest.json")

    loader = CorpusManifestLoader(manifest_path)

    manifest = loader.load_manifest()

    print("Corpus:")
    print(f"  ID: {manifest['corpus_id']}")
    print(f"  Version: {manifest['version']}")
    print(f"  Documents: {len(manifest['documents'])}")

    print("\nDocuments:")

    for entry in loader.load_document_entries():
        print(
            f"  - {entry['document_id']}: "
            f"{entry['title']} "
            f"({entry['source']})"
        )


if __name__ == "__main__":
    main()