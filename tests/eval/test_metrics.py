"""Tests for rag.eval.metrics — focused on capped_recall_at_k and summarize integration."""

from __future__ import annotations

from rag.eval.metrics import capped_recall_at_k, recall_at_k, summarize
from rag.eval.models import RetrievalResult


class TestCappedRecallAtK:
    """Unit tests for capped_recall_at_k."""

    def test_empty_relevant_returns_zero(self) -> None:
        assert capped_recall_at_k(["a", "b"], relevant=set(), k=5) == 0.0

    def test_equals_standard_recall_when_relevant_leq_k(self) -> None:
        """When |relevant| <= k, capped recall == standard recall."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c"}  # 2 relevant, k=5 → denominator is 2 either way
        assert capped_recall_at_k(retrieved, relevant, k=5) == recall_at_k(
            retrieved, relevant, k=5
        )

    def test_caps_denominator_when_relevant_exceeds_k(self) -> None:
        """The core fix: denominator is k, not |relevant|."""
        retrieved = ["a", "b", "c", "d", "e"]
        # 20 relevant chunks, only 2 hit in top-5
        relevant = {f"chunk-{i}" for i in range(20)} | {"a", "c"}

        standard = recall_at_k(retrieved, relevant, k=5)
        capped = capped_recall_at_k(retrieved, relevant, k=5)

        # Standard: 2/22 ≈ 0.09
        assert standard == 2 / 22
        # Capped: 2/5 = 0.4 — much more interpretable
        assert capped == 2 / 5

    def test_perfect_retrieval_capped(self) -> None:
        """All top-k slots filled with relevant chunks → 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = {f"chunk-{i}" for i in range(50)} | {"a", "b", "c"}
        assert capped_recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_no_hits(self) -> None:
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert capped_recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        """When retrieved list is shorter than k, only available items are checked."""
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c"}
        # k=10, but only 2 retrieved — 2 hits, denominator min(3, 10) = 3
        assert capped_recall_at_k(retrieved, relevant, k=10) == 2 / 3


class TestSummarizeIncludesCappedRecall:
    """Integration: summarize() populates capped_recall_at_k on RetrievalSummary."""

    def test_capped_recall_present_in_summary(self) -> None:
        results = [
            RetrievalResult(
                qid="q1",
                retrieved_chunk_ids=("a", "b", "c", "d", "e"),
                relevant_chunk_ids={"a", "c"},
            ),
        ]
        summary = summarize(results, ks=(5, 10))
        assert 5 in summary.capped_recall_at_k
        assert 10 in summary.capped_recall_at_k

    def test_capped_recall_in_flat_dict(self) -> None:
        results = [
            RetrievalResult(
                qid="q1",
                retrieved_chunk_ids=("a", "b"),
                relevant_chunk_ids={"a"},
            ),
        ]
        summary = summarize(results, ks=(10,))
        flat = summary.to_flat_dict()
        assert "capped_recall@10" in flat

    def test_capped_recall_round_trips_via_flat_dict(self) -> None:
        from rag.eval.models import RetrievalSummary

        results = [
            RetrievalResult(
                qid="q1",
                retrieved_chunk_ids=tuple(f"c{i}" for i in range(10)),
                relevant_chunk_ids={f"c{i}" for i in range(50)},
            ),
        ]
        summary = summarize(results, ks=(10,))
        flat = summary.to_flat_dict()
        restored = RetrievalSummary.from_flat_dict(flat)
        assert restored.capped_recall_at_k[10] == summary.capped_recall_at_k[10]
