from pathlib import Path

from src.rag.ingestion.html_extractor import (
    MedicalHTMLExtractor,
)


def main():
    html_path = Path(
        "data/rag/raw/AAD_MOLES_SYMPTOMS_001.html"
    )

    html = html_path.read_text(
        encoding="utf-8"
    )

    extractor = MedicalHTMLExtractor()

    text = extractor.extract(html)

    print(
        f"Extracted characters: {len(text)}"
    )

    print("\n" + "=" * 70)
    print("EXTRACTED MEDICAL TEXT")
    print("=" * 70)

    print(text[:10000])

    assert len(text) > 0

    # The extraction should contain article-related text.
    assert "Moles" in text or "mole" in text.lower()

    # Webpage implementation noise should not survive.
    assert "<script" not in text
    assert "<style" not in text

    print("\nHTML extraction test passed.")


if __name__ == "__main__":
    main()