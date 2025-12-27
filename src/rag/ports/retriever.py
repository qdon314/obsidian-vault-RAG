from __future__ import annotations

from typing import Mapping, Protocol

from rag.domain.models import Candidate


class Retriever(Protocol):
    """
    Retrieves candidate chunks for a query string.
    """

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Candidate]:
        ...
