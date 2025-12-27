from __future__ import annotations

from typing import Protocol

from rag.domain.models import QueryTrace


class QueryLogger(Protocol):
    """
    Persists query traces (typically JSONL). Useful for debugging + eval.
    """

    def log(self, trace: QueryTrace) -> None:
        ...
