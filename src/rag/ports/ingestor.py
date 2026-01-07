from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from rag.domain.models import Document, IngestReport


class Ingestor(Protocol):
    """
    Converts raw inputs (paths, URLs, etc.) into Documents.
    """

    def ingest(self, inputs: Sequence[str], *, metadata: Mapping[str, object] | None = None) -> tuple[list[Document], IngestReport]:
        ...
