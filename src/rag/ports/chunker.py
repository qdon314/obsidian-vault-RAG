from __future__ import annotations

from typing import Mapping, Protocol

from rag.domain.models import Chunk, Document


class Chunker(Protocol):
    """
    Splits a Document into Chunks.
    """

    def chunk(self, doc: Document, *, metadata: Mapping[str, object] | None = None) -> list[Chunk]:
        ...
