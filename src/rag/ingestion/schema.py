from dataclasses import dataclass, field
from typing import Any


@dataclass
class MedicalDocument:
    """
    Canonical representation of a medical knowledge document.

    A MedicalDocument represents a complete source document before
    chunking and embedding.
    """

    document_id: str
    title: str
    source: str
    source_url: str | None
    text: str

    topic: str | None = None
    condition: str | None = None
    sections: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)