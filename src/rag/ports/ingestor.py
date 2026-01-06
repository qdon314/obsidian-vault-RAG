from __future__ import annotations

from typing import Mapping, Protocol, Sequence, Tuple

from rag.domain.models import Document, IngestReport

class Ingestor(Protocol):
    """
    Converts raw inputs (paths, URLs, etc.) into Documents.
    """

    def ingest(self, inputs: Sequence[str], *, metadata: Mapping[str, object] | None = None) -> Tuple[list[Document], IngestReport]:
        ...
