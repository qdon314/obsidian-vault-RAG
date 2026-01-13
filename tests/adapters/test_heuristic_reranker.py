"""Tests for HeuristicReranker adapter."""

from __future__ import annotations

from rag.adapters.reranking.rerank_heuristic import HeuristicReranker
from rag.ports import Reranker
from tests.conftest import make_candidate, make_chunk


class TestHeuristicRerankerBasics:
    """Basic reranking behavior."""

    def test_rerank_empty_candidates(self, heuristic_reranker: Reranker):
        """Reranking empty list returns empty list."""
        result = heuristic_reranker.rerank("test query", [])

        assert result == []

    def test_rerank_returns_candidates(self, heuristic_reranker: Reranker):
        """Rerank returns Candidate objects."""
        candidates = [make_candidate(score=0.8)]

        result = heuristic_reranker.rerank("query", candidates)

        assert len(result) == 1
        assert hasattr(result[0], "chunk")
        assert hasattr(result[0], "score")

    def test_rerank_preserves_candidates_without_diversify(self):
        """Without diversification, all candidates are returned."""
        reranker = HeuristicReranker(diversify=False)
        candidates = [
            make_candidate(score=0.8),
            make_candidate(score=0.6),
            make_candidate(score=0.9),
        ]

        result = reranker.rerank("query", candidates)

        assert len(result) == 3

    def test_name_property(self, heuristic_reranker: Reranker):
        """Reranker exposes its name."""
        assert heuristic_reranker.name == "heuristic_v1"


class TestHeuristicRerankerOverlap:
    """Lexical overlap scoring."""

    def test_overlap_boosts_matching_candidates(self):
        """Candidates with query token overlap get boosted."""
        reranker = HeuristicReranker(overlap_weight=1.0, diversify=False)

        # Candidate with no overlap
        chunk_no_match = make_chunk(
            chunk_id="no-match",
            text="completely different unrelated content",
        )
        cand_no_match = make_candidate(chunk=chunk_no_match, score=0.5)

        # Candidate with overlap
        chunk_match = make_chunk(
            chunk_id="match",
            text="machine learning is great for learning",
        )
        cand_match = make_candidate(chunk=chunk_match, score=0.5)

        candidates = [cand_no_match, cand_match]
        query = "machine learning"

        result = reranker.rerank(query, candidates)

        # The matching candidate should be ranked higher
        assert result[0].chunk.chunk_id == "match"

    def test_zero_overlap_weight_ignores_overlap(self):
        """With overlap_weight=0, only base scores matter."""
        reranker = HeuristicReranker(overlap_weight=0.0, diversify=False)

        chunk_overlap = make_chunk(chunk_id="overlap", text="python python python")
        chunk_no_overlap = make_chunk(chunk_id="no-overlap", text="rust java go")

        candidates = [
            make_candidate(chunk=chunk_overlap, score=0.3),
            make_candidate(chunk=chunk_no_overlap, score=0.9),
        ]

        result = reranker.rerank("python", candidates)

        # Higher base score should win
        assert result[0].chunk.chunk_id == "no-overlap"

    def test_high_base_score_can_beat_overlap(self):
        """High base score can still beat overlap boost."""
        reranker = HeuristicReranker(overlap_weight=0.15, diversify=False)

        low_score_overlap = make_candidate(
            chunk=make_chunk(chunk_id="low", text="python python"),
            score=0.3,
        )
        high_score_no_overlap = make_candidate(
            chunk=make_chunk(chunk_id="high", text="rust java typescript"),
            score=0.9,
        )

        result = reranker.rerank("python", [low_score_overlap, high_score_no_overlap])

        # High base score should win with modest overlap weight
        assert result[0].chunk.chunk_id == "high"


class TestHeuristicRerankerDiversification:
    """Document diversification."""

    def test_diversify_limits_per_doc(self):
        """Diversification limits chunks from same doc_id."""
        reranker = HeuristicReranker(diversify=True, max_per_doc=2)

        # Create 5 chunks from same doc
        candidates = [
            make_candidate(
                chunk=make_chunk(
                    chunk_id=f"doc1:chunk{i}",
                    metadata={"doc_id": "doc1"},
                ),
                score=0.9 - i * 0.01,
            )
            for i in range(5)
        ]

        result = reranker.rerank("test", candidates)

        assert len(result) == 2

    def test_diversify_across_multiple_docs(self):
        """Diversification allows chunks from different docs."""
        reranker = HeuristicReranker(diversify=True, max_per_doc=1)

        candidates = [
            make_candidate(
                chunk=make_chunk(chunk_id="a", metadata={"doc_id": "doc1"}),
                score=0.95,
            ),
            make_candidate(
                chunk=make_chunk(chunk_id="b", metadata={"doc_id": "doc1"}),
                score=0.94,
            ),
            make_candidate(
                chunk=make_chunk(chunk_id="c", metadata={"doc_id": "doc2"}),
                score=0.93,
            ),
            make_candidate(
                chunk=make_chunk(chunk_id="d", metadata={"doc_id": "doc3"}),
                score=0.92,
            ),
        ]

        result = reranker.rerank("test", candidates)

        # Should get one from each doc
        assert len(result) == 3
        doc_ids = [c.chunk.metadata.get("doc_id") for c in result]
        assert len(set(doc_ids)) == 3

    def test_diversify_uses_path_fallback(self):
        """Diversification uses 'path' if 'doc_id' not in metadata."""
        reranker = HeuristicReranker(diversify=True, max_per_doc=1)

        candidates = [
            make_candidate(
                chunk=make_chunk(chunk_id="a", metadata={"path": "/doc1.md"}),
                score=0.9,
            ),
            make_candidate(
                chunk=make_chunk(chunk_id="b", metadata={"path": "/doc1.md"}),
                score=0.8,
            ),
            make_candidate(
                chunk=make_chunk(chunk_id="c", metadata={"path": "/doc2.md"}),
                score=0.7,
            ),
        ]

        result = reranker.rerank("test", candidates)

        # Should limit by path
        assert len(result) == 2

    def test_no_diversify_keeps_all(self):
        """Without diversification, all chunks from same doc are kept."""
        reranker = HeuristicReranker(diversify=False)

        candidates = [
            make_candidate(
                chunk=make_chunk(chunk_id=f"doc1:chunk{i}", metadata={"doc_id": "doc1"}),
                score=0.9 - i * 0.01,
            )
            for i in range(5)
        ]

        result = reranker.rerank("test", candidates)

        assert len(result) == 5
