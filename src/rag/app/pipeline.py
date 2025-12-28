from __future__ import annotations

from rag.domain.models import Document
from rag.ports import Chunker


def chunk_document(doc: Document, *, chunker: Chunker) -> int:
    """
    Example pipeline step that depends only on the Chunker interface (port).
    """
    chunks = chunker.chunk(doc)
    # for now, just return how many chunks we created
    return len(chunks)
