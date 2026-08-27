import re


class TextCleaner:
    """
    Cleans extracted medical text while preserving meaningful content.
    """

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""

        # Normalize line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove null characters and other control characters.
        text = text.replace("\x00", "")

        # Normalize horizontal whitespace.
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse excessive blank lines.
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Remove leading/trailing whitespace from lines.
        lines = [line.strip() for line in text.splitlines()]

        text = "\n".join(lines)

        # Final cleanup.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()