import json
from collections.abc import Iterable
from pathlib import Path


INDEX_PATH = Path("data/rag/indexes/medical_v0.1/chunks.json")


CLASS_TERMS = {
    "ACK": [
        "actinic keratosis",
        "actinic keratoses",
    ],
    "BCC": [
        "basal cell carcinoma",
    ],
    "MEL": [
        "melanoma",
    ],
    "NEV": [
        "nevus",
        "nevi",
    ],
    "SCC": [
        "squamous cell carcinoma",
    ],
    "SEK": [
        "seborrheic keratosis",
        "seborrheic keratoses",
    ],
}


def load_documents(chunks: Iterable[dict]) -> dict[str, dict[str, str]]:
    """Combine chunks by document, preserving source-level provenance."""
    documents = {}

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        document_id = chunk.get("document_id", "UNKNOWN")
        title = metadata.get("title", "UNKNOWN")
        text = chunk.get("text", "")

        if document_id not in documents:
            documents[document_id] = {
                "title": title,
                "text": "",
            }

        documents[document_id]["text"] += "\n" + text

    return documents


def find_class_coverage(
    documents: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    """Return matching document IDs for every CV-native class."""
    coverage = {}

    for class_name, terms in CLASS_TERMS.items():
        coverage[class_name] = [
            document_id
            for document_id, document in documents.items()
            if any(
                term.lower() in document["text"].lower()
                for term in terms
            )
        ]

    return coverage


def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index chunks not found: {INDEX_PATH}")

    chunks = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    documents = load_documents(chunks)
    coverage = find_class_coverage(documents)

    print("\n=== CORPUS DOCUMENTS ===\n")

    for document_id, doc in documents.items():
        print(f"{document_id}")
        print(f"  {doc['title']}")

    print("\n=== CV CLASS COVERAGE ===\n")

    for class_name, matches in coverage.items():
        if matches:
            print(f"{class_name}: COVERED")
            for document_id in matches:
                print(f"  - {document_id}: {documents[document_id]['title']}")
        else:
            print(f"{class_name}: NOT COVERED")

    print()


if __name__ == "__main__":
    main()
