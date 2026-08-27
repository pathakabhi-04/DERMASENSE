from src.rag.ingestion.cleaner import TextCleaner


def main():
    raw_text = """
    Melanoma   is a type of skin cancer.


    A changing mole may require
    professional evaluation.
    """

    cleaned = TextCleaner.clean(raw_text)

    print("RAW:")
    print(repr(raw_text))

    print("\nCLEANED:")
    print(repr(cleaned))


if __name__ == "__main__":
    main()