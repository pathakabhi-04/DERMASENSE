from pathlib import Path

from src.rag.ingestion.acquisition import (
    acquire_corpus,
    save_acquisition_manifest,
)


def main():
    manifest_path = Path(
        "data/rag/corpus_manifest.json"
    )

    raw_dir = Path(
        "data/rag/raw"
    )

    acquisition_manifest = Path(
        "data/rag/acquisition_manifest.json"
    )

    records = acquire_corpus(
        manifest_path=manifest_path,
        output_dir=raw_dir,
    )

    save_acquisition_manifest(
        records=records,
        output_path=acquisition_manifest,
    )

    print(
        f"\nAcquired {len(records)} documents."
    )

    print(
        f"Acquisition manifest: "
        f"{acquisition_manifest}"
    )


if __name__ == "__main__":
    main()