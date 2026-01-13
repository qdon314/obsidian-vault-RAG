from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from rag.domain.filters import Where
from rag.domain.models import Candidate

Vector = list[float]

class Retriever(Protocol):
    """
    Retrieves candidate chunks for a query string.
    """

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        where: Where = None,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Candidate]:
        ...
