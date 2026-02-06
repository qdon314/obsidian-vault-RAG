"""Tests for BM25Retriever: keyword retrieval via BM25 scoring."""

from __future__ import annotations

import pytest

from rag.adapters.retrieval.bm25_retriever import BM25Retriever
from rag.domain.errors import RetrievalError
from rag.domain.filters import Eq
from tests.conftest import make_chunk


def _make_corpus() -> list:
    return [
        make_chunk(
            chunk_id="c1",
            doc_id="d1",
            text="Python is a programming language used for data science",
            metadata={"tag": "python"},
        ),
        make_chunk(
            chunk_id="c2",
            doc_id="d1",
            text="Machine learning uses statistical methods and algorithms",
            metadata={"tag": "ml"},
        ),
        make_chunk(
            chunk_id="c3",
            doc_id="d2",
            text="Vector databases store embeddings for similarity search",
            metadata={"tag": "infra"},
        ),
        make_chunk(
            chunk_id="c4",
            doc_id="d2",
            text="BM25 is a keyword ranking algorithm used in information retrieval",
            metadata={"tag": "infra"},
        ),
    ]


class TestBM25FindsExactTerms:
    """BM25 retriever finds chunks containing exact query terms."""

    def test_exact_term_match(self) -> None:
        retriever = BM25Retriever()
        retriever.index(_make_corpus())

        results = retriever.retrieve("Python programming", top_k=2)

        assert len(results) >= 1
        assert results[0].chunk.chunk_id == "c1"

    def test_rare_term_ranked_higher(self) -> None:
        retriever = BM25Retriever()
        retriever.index(_make_corpus())

        results = retriever.retrieve("BM25 algorithm", top_k=4)

        assert results[0].chunk.chunk_id == "c4"

    def test_no_match_returns_empty(self) -> None:
        retriever = BM25Retriever()
        retriever.index(_make_corpus())

        results = retriever.retrieve("quantum entanglement", top_k=5)

        assert results == []


class TestBM25Scoring:
    """BM25 scores are positive and ordered correctly."""

    def test_scores_are_positive(self) -> None:
        retriever = BM25Retriever()
        retriever.index(_make_corpus())

        results = retriever.retrieve("machine learning algorithms", top_k=4)

        for cand in results:
            assert cand.score > 0

    def test_results_sorted_by_score_descending(self) -> None:
        retriever = BM25Retriever()
        retriever.index(_make_corpus())

        results = retriever.retrieve("data science statistical methods", top_k=4)

        scores = [c.score for c in results]
        assert scores == sorted(scores, reverse=True)


class TestBM25Filtering:
    """BM25 respects the where filter parameter."""

    def test_filter_restricts_results(self) -> None:
        retriever = BM25Retriever()
        retriever.index(_make_corpus())

        results = retriever.retrieve(
            "algorithm",
            top_k=4,
            where=Eq(field="tag", value="infra"),
        )

        assert all(c.chunk.metadata["tag"] == "infra" for c in results)

    def test_filter_none_returns_all_matches(self) -> None:
        retriever = BM25Retriever()
        retriever.index(_make_corpus())

        results = retriever.retrieve("is a", top_k=10, where=None)

        assert len(results) >= 2


class TestBM25LazyIndexing:
    """Lazy indexing via chunk_source callback."""

    def test_lazy_index_on_first_retrieve(self) -> None:
        corpus = _make_corpus()
        retriever = BM25Retriever(chunk_source=lambda: corpus)

        results = retriever.retrieve("Python", top_k=2)

        assert len(results) >= 1
        assert results[0].chunk.chunk_id == "c1"

    def test_no_index_no_source_raises(self) -> None:
        retriever = BM25Retriever()

        with pytest.raises(RetrievalError, match="no index"):
            retriever.retrieve("anything", top_k=5)


class TestBM25EdgeCases:
    """Edge cases and protocol conformance."""

    def test_empty_corpus(self) -> None:
        retriever = BM25Retriever()
        retriever.index([])

        results = retriever.retrieve("test", top_k=5)

        assert results == []

    def test_single_chunk_corpus(self) -> None:
        retriever = BM25Retriever()
        retriever.index([make_chunk(text="hello world")])

        results = retriever.retrieve("hello", top_k=5)

        assert len(results) == 1

    def test_top_k_limits_results(self) -> None:
        retriever = BM25Retriever()
        retriever.index(_make_corpus())

        results = retriever.retrieve("is a", top_k=1)

        assert len(results) <= 1

    def test_custom_k1_b_parameters(self) -> None:
        retriever = BM25Retriever(k1=2.0, b=0.5)
        retriever.index(_make_corpus())

        results = retriever.retrieve("Python", top_k=2)

        assert len(results) >= 1
