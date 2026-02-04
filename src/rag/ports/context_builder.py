from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from rag.domain.models import Candidate, ContextPack


class ContextBuilder(Protocol):
    """
    Takes candidates and constructs the final prompt context within a token budget.

    The caller is responsible for ordering candidates (e.g., by rerank score).
    This component only packs them into context - it has no awareness of reranking.
    """

    def build(
        self,
        query: str,
        candidates: Sequence[Candidate],
        *,
        token_budget: int,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextPack: ...
