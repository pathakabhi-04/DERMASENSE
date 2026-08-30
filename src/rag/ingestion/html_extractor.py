from __future__ import annotations

from bs4 import BeautifulSoup


class HTMLExtractionError(RuntimeError):
    """Raised when meaningful article content cannot be extracted."""


class MedicalHTMLExtractor:
    """
    Extract readable medical article content from a webpage snapshot.

    The extractor removes executable/presentation elements and keeps
    headings, paragraphs, and list items from the main article content.
    """

    REMOVE_TAGS = (
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
        "form",
        "nav",
        "footer",
        "header",
    )

    CONTENT_SELECTORS = (
        "article",
        "main",
        '[role="main"]',
    )

    def extract(self, html: str) -> str:
        """
        Convert raw HTML into readable structured text.
        """

        if not html.strip():
            raise HTMLExtractionError(
                "HTML content is empty."
            )

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        # Remove webpage implementation noise.
        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Locate the main article content.
        content = self._find_content_root(
            soup
        )

        if content is None:
            raise HTMLExtractionError(
                "Could not identify main article content."
            )

        # Preserve semantic document structure.
        elements = content.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "p",
                "li",
            ]
        )

        blocks: list[str] = []

        for element in elements:
            text = element.get_text(
                " ",
                strip=True,
            )

            if not text:
                continue

            # Remove only exact adjacent duplicates.
            if blocks and blocks[-1] == text:
                continue

            blocks.append(text)

        if not blocks:
            raise HTMLExtractionError(
                "No meaningful text found in article content."
            )

        return "\n\n".join(blocks)

    def _find_content_root(self, soup):
        """
        Find the most likely main-content container.
        """

        for selector in self.CONTENT_SELECTORS:
            element = soup.select_one(
                selector
            )

            if element is not None:
                return element

        return None