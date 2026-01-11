from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from rag.domain.models import Chunk


class ChunkLoader(Protocol):
    """
    Loads chunks from a storage backend.
    """

    def load_all(
        self,
        path: Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Chunk]:
        """Load all chunks from the given path."""
        ...

    def load_by_doc_id(
        self,
        path: Path,
        doc_id: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Chunk]:
        """Load chunks for a specific document."""
        ...

    def get_doc_ids(
        self,
        path: Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> list[str]:
        """Get all unique document IDs."""
        ...
