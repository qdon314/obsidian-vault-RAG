from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from rag.domain.models import Chunk, Document


class Chunker(Protocol):
    """
    Splits a Document into Chunks.
    """

    def chunk(
        self, doc: Document, *, metadata: Mapping[str, object] | None = None
    ) -> list[Chunk]: ...

    def get_config(self) -> dict[str, Any]:
        """Return chunker configuration for manifest/introspection."""
        ...
