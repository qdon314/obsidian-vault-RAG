from __future__ import annotations
from rag.ports import Reranker
from rag.domain.models import Candidate

class NoOpReranker(Reranker):
    @property
    def name(self) -> str:
        return "noop"

    def rerank(self, query: str, candidates: list[Candidate], *, metadata=None) -> list[Candidate]:
        return list(candidates)
