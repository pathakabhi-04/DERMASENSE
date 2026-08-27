from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """
    A retrieval unit produced from a MedicalDocument.
    """

    chunk_id: str
    document_id: str

    text: str

    chunk_index: int
    start_char: int
    end_char: int

    metadata: dict[str, Any] = field(default_factory=dict)