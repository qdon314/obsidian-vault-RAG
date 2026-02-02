from __future__ import annotations

from rag.domain.models import Candidate


class NoOpReranker:
    @property
    def name(self) -> str:
        return "noop"

    def rerank(self, query: str, candidates: list[Candidate], *, metadata=None) -> list[Candidate]:
        return list(candidates)
