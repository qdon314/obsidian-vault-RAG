from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rag.domain.filters import Where
from rag.domain.models import Candidate
from rag.ports import Embedder, VectorStore


@dataclass(frozen=True, slots=True)
class VectorRetriever:
    """
    Retrieves candidates by embedding the query and searching the vector store.
    """
    embedder: Embedder
    store: VectorStore

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        where: Where = None,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Candidate]:
        q_vec = self.embedder.embed_texts([query], metadata=metadata)[0]
        return self.store.search(query_vector=q_vec, top_k=top_k, where=where, metadata=metadata)