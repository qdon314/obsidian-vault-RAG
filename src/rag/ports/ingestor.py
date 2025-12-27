from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from rag.domain.models import Document


class Ingestor(Protocol):
    """
    Converts raw inputs (paths, URLs, etc.) into Documents.
    """

    def ingest(self, inputs: Sequence[str], *, metadata: Mapping[str, object] | None = None) -> list[Document]:
        ...
