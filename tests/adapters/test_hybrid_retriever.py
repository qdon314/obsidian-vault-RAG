"""Tests for HybridRetriever: RRF fusion of two retrievers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rag.adapters.retrieval.hybrid_retriever import HybridRetriever
from rag.domain.filters import Where
from rag.domain.models import Candidate
from tests.conftest import make_candidate, make_chunk

# ---------------------------------------------------------------------------
# Stub retriever for controlled test inputs
# ---------------------------------------------------------------------------


@dataclass
class StubRetriever:
    """Returns a fixed list of candidates, ignoring the query."""

    results: list[Candidate]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        where: Where = None,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Candidate]:
        return self.results[:top_k]


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

_CHUNK_A = make_chunk(chunk_id="a", doc_id="d1", text="chunk A")
_CHUNK_B = make_chunk(chunk_id="b", doc_id="d1", text="chunk B")
_CHUNK_C = make_chunk(chunk_id="c", doc_id="d2", text="chunk C")
_CHUNK_D = make_chunk(chunk_id="d", doc_id="d2", text="chunk D")


class TestRRFOverlapRankedHighest:
    """Items appearing in both result lists get highest fused scores."""

    def test_overlap_item_ranked_first(self) -> None:
        # chunk B appears in both lists; A only in primary, C only in secondary
        primary = StubRetriever(
            results=[
                make_candidate(chunk=_CHUNK_A, score=0.9),
                make_candidate(chunk=_CHUNK_B, score=0.8),
            ]
        )
        secondary = StubRetriever(
            results=[
                make_candidate(chunk=_CHUNK_B, score=5.0),
                make_candidate(chunk=_CHUNK_C, score=3.0),
            ]
        )
        hybrid = HybridRetriever(primary=primary, secondary=secondary)

        results = hybrid.retrieve("query", top_k=3)

        assert results[0].chunk.chunk_id == "b"

    def test_overlap_score_higher_than_single_list(self) -> None:
        primary = StubRetriever(
            results=[
                make_candidate(chunk=_CHUNK_A, score=0.9),
                make_candidate(chunk=_CHUNK_B, score=0.8),
            ]
        )
        secondary = StubRetriever(
            results=[
                make_candidate(chunk=_CHUNK_B, score=5.0),
                make_candidate(chunk=_CHUNK_C, score=3.0),
            ]
        )
        hybrid = HybridRetriever(primary=primary, secondary=secondary)

        results = hybrid.retrieve("query", top_k=3)

        overlap_score = results[0].score  # chunk B (in both)
        single_scores = [r.score for r in results[1:]]
        assert all(overlap_score > s for s in single_scores)


class TestRRFWeights:
    """Primary/secondary weights affect ranking."""

    def test_primary_weighted_higher_by_default(self) -> None:
        # A is rank 1 in primary, B is rank 1 in secondary
        primary = StubRetriever(results=[make_candidate(chunk=_CHUNK_A, score=0.9)])
        secondary = StubRetriever(results=[make_candidate(chunk=_CHUNK_B, score=5.0)])
        hybrid = HybridRetriever(
            primary=primary,
            secondary=secondary,
            primary_weight=0.7,
            secondary_weight=0.3,
        )

        results = hybrid.retrieve("query", top_k=2)

        assert results[0].chunk.chunk_id == "a"

    def test_equal_weights_symmetric(self) -> None:
        primary = StubRetriever(results=[make_candidate(chunk=_CHUNK_A, score=0.9)])
        secondary = StubRetriever(results=[make_candidate(chunk=_CHUNK_B, score=5.0)])
        hybrid = HybridRetriever(
            primary=primary,
            secondary=secondary,
            primary_weight=1.0,
            secondary_weight=1.0,
        )

        results = hybrid.retrieve("query", top_k=2)

        # Same weight + same rank → equal scores
        assert abs(results[0].score - results[1].score) < 1e-9


class TestRRFTopK:
    """Top-k limits and empty inputs."""

    def test_top_k_limits_output(self) -> None:
        primary = StubRetriever(
            results=[
                make_candidate(chunk=_CHUNK_A, score=0.9),
                make_candidate(chunk=_CHUNK_B, score=0.8),
            ]
        )
        secondary = StubRetriever(
            results=[
                make_candidate(chunk=_CHUNK_C, score=5.0),
                make_candidate(chunk=_CHUNK_D, score=3.0),
            ]
        )
        hybrid = HybridRetriever(primary=primary, secondary=secondary)

        results = hybrid.retrieve("query", top_k=2)

        assert len(results) == 2

    def test_empty_secondary_uses_primary_only(self) -> None:
        primary = StubRetriever(results=[make_candidate(chunk=_CHUNK_A, score=0.9)])
        secondary = StubRetriever(results=[])
        hybrid = HybridRetriever(primary=primary, secondary=secondary)

        results = hybrid.retrieve("query", top_k=5)

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "a"

    def test_both_empty_returns_empty(self) -> None:
        hybrid = HybridRetriever(
            primary=StubRetriever(results=[]),
            secondary=StubRetriever(results=[]),
        )

        results = hybrid.retrieve("query", top_k=5)

        assert results == []


class TestHybridRecallOnKeywordQueries:
    """Hybrid recall >= vector-only recall on keyword/acronym queries.

    TODO: Replace the StubRetriever standing in for vector search with
    a real VectorRetriever + DummyEmbedder + InMemoryVectorStore so
    this becomes a true integration test.  Currently the test verifies
    the structural guarantee (BM25 finds keyword hits that the stub
    "vector" retriever misses, and RRF surfaces them) without involving
    real embeddings.
    """

    def test_hybrid_recall_ge_vector_on_keyword_queries(self) -> None:
        from rag.adapters.retrieval.bm25_retriever import BM25Retriever

        # Corpus with a rare acronym "ALARA" that vector search (stub) misses
        corpus = [
            make_chunk(
                chunk_id="c1", doc_id="d1", text="Radiation safety follows ALARA principles"
            ),
            make_chunk(chunk_id="c2", doc_id="d1", text="Dose limits are set by regulatory bodies"),
            make_chunk(chunk_id="c3", doc_id="d2", text="Shielding materials reduce exposure"),
        ]

        # Stub vector retriever: returns c2, c3 (misses the ALARA chunk)
        vector_stub = StubRetriever(
            results=[
                make_candidate(chunk=corpus[1], score=0.8),
                make_candidate(chunk=corpus[2], score=0.7),
            ]
        )
        vector_chunk_ids = {c.chunk.chunk_id for c in vector_stub.results}

        # BM25 retriever: will find "ALARA" via exact keyword match
        bm25 = BM25Retriever()
        bm25.index(corpus)

        hybrid = HybridRetriever(primary=vector_stub, secondary=bm25)
        hybrid_results = hybrid.retrieve("ALARA", top_k=5)
        hybrid_chunk_ids = {c.chunk.chunk_id for c in hybrid_results}

        # Hybrid must include everything vector found, plus the keyword hit
        assert vector_chunk_ids <= hybrid_chunk_ids
        assert "c1" in hybrid_chunk_ids


class TestVectorOnlyDefault:
    """Default config uses vector-only retriever, not hybrid."""

    def test_default_backend_is_vector(self) -> None:
        from rag.settings import Retrieval

        r = Retrieval()
        assert r.backend == "vector"
