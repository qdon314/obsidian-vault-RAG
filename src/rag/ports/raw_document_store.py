"""Port for storing and retrieving raw documents in the corpus-of-record."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag.domain.models import Document


@runtime_checkable
class RawDocumentStore(Protocol):
    """Stores and retrieves raw (pre-chunking) documents.

    Implementations persist normalized document JSON to durable storage
    (S3) so the corpus-of-record is remote and not tied to local disk.
    """

    def store_document(
        self,
        doc: Document,
        *,
        corpus_id: str,
        content_sha256: str,
    ) -> str:
        """Persist a raw document. Returns the storage key (e.g. S3 key)."""
        ...

    def get_document(self, key: str) -> Document:
        """Retrieve a raw document by storage key."""
        ...
