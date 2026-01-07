from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from rag.domain.models import Candidate


class Reranker(Protocol):
    """
    Re-orders candidates based on relevance to the query.
    """

    @property
    def name(self) -> str: ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Candidate]:
        ...
