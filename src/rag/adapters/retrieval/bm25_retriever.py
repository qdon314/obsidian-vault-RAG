from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from rag.adapters.filters.inmemory_evaluator import InMemoryFilterEvaluator
from rag.domain.errors import RetrievalError
from rag.domain.filters import Where
from rag.domain.models import Candidate, Chunk


@dataclass(slots=True)
class BM25Retriever:
    """BM25 keyword retriever implementing the Retriever port.

    Operates over in-memory chunks with no external dependencies.
    Supports lazy indexing via an optional *chunk_source* callback:
    on the first ``retrieve()`` call the callback is invoked and the
    index is built automatically.
    """

    k1: float = 1.5
    b: float = 0.75
    chunk_source: Callable[[], Sequence[Chunk]] | None = None

    # Internal index state (built by ``index()``)
    _chunks: list[Chunk] = field(default_factory=list, init=False, repr=False)
    _tokenized: list[list[str]] = field(default_factory=list, init=False, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _avg_dl: float = field(default=0.0, init=False, repr=False)
    _indexed: bool = field(default=False, init=False, repr=False)
    _filter_eval: InMemoryFilterEvaluator = field(
        default_factory=InMemoryFilterEvaluator, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build the BM25 index from *chunks*."""
        self._chunks = list(chunks)
        self._tokenized = [self._tokenize(c.text) for c in self._chunks]
        n = len(self._tokenized)
        self._avg_dl = sum(len(t) for t in self._tokenized) / max(n, 1)
        self._idf = self._compute_idf()
        self._indexed = True

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        where: Where = None,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Candidate]:
        self._ensure_indexed()
        query_tokens = self._tokenize(query)
        scored: list[tuple[int, float]] = []
        for i, doc_tokens in enumerate(self._tokenized):
            if where is not None:
                record = Chunk.to_record(self._chunks[i])
                if not self._filter_eval.matches(where, record):
                    continue
            s = self._score(query_tokens, doc_tokens)
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [Candidate(chunk=self._chunks[i], score=score) for i, score in scored[:top_k]]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_indexed(self) -> None:
        if self._indexed:
            return
        if self.chunk_source is None:
            raise RetrievalError(
                "BM25Retriever has no index. Call index() or provide a chunk_source."
            )
        self.index(self.chunk_source())

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase whitespace split. Intentionally simple."""
        return text.lower().split()

    def _compute_idf(self) -> dict[str, float]:
        n = len(self._tokenized)
        df: dict[str, int] = {}
        for tokens in self._tokenized:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        return {term: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()}

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        dl = len(doc_tokens)
        tf: dict[str, int] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for qt in query_tokens:
            if qt not in self._idf:
                continue
            f = tf.get(qt, 0)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
            score += self._idf[qt] * numerator / denominator
        return score
