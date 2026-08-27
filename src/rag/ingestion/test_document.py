from pathlib import Path

from src.rag.ingestion.cleaner import TextCleaner
from src.rag.ingestion.loader import (
    CorpusManifestLoader,
    LocalTextLoader,
)


def main():
    manifest_path = Path("data/rag/corpus_manifest.json")
    source_path = Path("data/rag/raw/test/sample.txt")

    manifest_loader = CorpusManifestLoader(manifest_path)
    text_loader = LocalTextLoader()

    entries = manifest_loader.load_document_entries()

    print(f"Manifest documents: {len(entries)}")

    raw_text = text_loader.load(source_path)
    cleaned_text = TextCleaner.clean(raw_text)

    print("\nCleaned source:")
    print(cleaned_text)


if __name__ == "__main__":
    main()