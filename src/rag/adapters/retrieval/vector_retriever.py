from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from rag.ports import Embedder, VectorStore
from rag.domain.models import Candidate


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
        filters: Optional[Mapping[str, object]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> list[Candidate]:
        q_vec = self.embedder.embed_texts([query], metadata=metadata)[0]
        return self.store.search(query_vector=q_vec, top_k=top_k, filters=filters, metadata=metadata)
