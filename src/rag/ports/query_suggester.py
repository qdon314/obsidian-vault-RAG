from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from rag.domain.models import Chunk
from rag.eval.schema import QuerySuggestion


class QuerySuggester(Protocol):
    """
    Generates query suggestions from chunk content using an LLM.
    """

    def suggest_queries(
        self,
        chunk: Chunk,
        num_suggestions: int = 3,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> list[QuerySuggestion]:
        """Generate query suggestions based on chunk content."""
        ...
