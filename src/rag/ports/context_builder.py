from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from rag.domain.models import Candidate, ContextPack


class ContextBuilder(Protocol):
    """
    Takes candidates and constructs the final prompt context within a token budget.
    """

    def build(
        self,
        query: str,
        candidates: Sequence[Candidate],
        *,
        token_budget: int,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextPack:
        ...
