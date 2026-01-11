from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from rag.eval.schema import EvalQuery


class EvalStore(Protocol):
    """
    Persists and loads evaluation queries.
    """

    def load_queries(
        self,
        path: Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> list[EvalQuery]:
        """Load all queries from the given path."""
        ...

    def save_queries(
        self,
        queries: list[EvalQuery],
        path: Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Save all queries to the given path (overwrites)."""
        ...

    def append_query(
        self,
        query: EvalQuery,
        path: Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Append a single query to the file."""
        ...

    def count_queries(
        self,
        path: Path,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> int:
        """Count queries in the file without loading all."""
        ...
