"""Tests for query diff logic (spec 06)."""

from __future__ import annotations

import pytest

from eval.app.results.query_diff import compute_retrieval_diff, natural_sort_key
from rag.eval.models import EvalResult, RetrievalResult


class TestNaturalSortKey:
    def test_numeric_ordering(self) -> None:
        """q_2 sorts before q_10."""
        qids = ["q_10", "q_2", "q_100", "q_1"]
        result = sorted(qids, key=natural_sort_key)
        assert result == ["q_1", "q_2", "q_10", "q_100"]

    def test_mixed_alpha_numeric(self) -> None:
        """Handles mixed prefixes with numeric suffixes."""
        qids = ["a_2", "b_1", "a_10", "a_1"]
        result = sorted(qids, key=natural_sort_key)
        assert result == ["a_1", "a_2", "a_10", "b_1"]

    def test_pure_numeric(self) -> None:
        """Handles bare numbers."""
        qids = ["10", "2", "1"]
        result = sorted(qids, key=natural_sort_key)
        assert result == ["1", "2", "10"]

    def test_no_numbers(self) -> None:
        """Falls back to lexicographic for non-numeric strings."""
        qids = ["beta", "alpha", "gamma"]
        result = sorted(qids, key=natural_sort_key)
        assert result == ["alpha", "beta", "gamma"]


def _make_result(
    qid: str,
    retrieved: list[str],
    relevant: set[str],
) -> EvalResult:
    """Helper to create a minimal EvalResult for testing."""
    return EvalResult(
        qid=qid,
        query="test query",
        retrieval_result=RetrievalResult(
            qid=qid,
            retrieved_chunk_ids=tuple(retrieved),
            relevant_chunk_ids=relevant,
        ),
    )


class TestComputeRetrievalDiff:
    def test_tp_lost(self) -> None:
        """Relevant chunk in A but not B -> TP lost."""
        a = _make_result("q1", ["c1", "c2"], {"c1"})
        b = _make_result("q1", ["c2", "c3"], {"c1"})
        rows = compute_retrieval_diff(a, b)

        c1_row = next(r for r in rows if r.chunk_id == "c1")
        assert c1_row.relevant is True
        assert c1_row.rank_a == 1
        assert c1_row.rank_b is None
        assert c1_row.status == "TP lost"

    def test_tp_gained(self) -> None:
        """Relevant chunk in B but not A -> TP gained."""
        a = _make_result("q1", ["c2"], {"c1"})
        b = _make_result("q1", ["c1", "c2"], {"c1"})
        rows = compute_retrieval_diff(a, b)

        c1_row = next(r for r in rows if r.chunk_id == "c1")
        assert c1_row.relevant is True
        assert c1_row.rank_a is None
        assert c1_row.rank_b == 1
        assert c1_row.status == "TP gained"

    def test_fp_lost(self) -> None:
        """Irrelevant chunk in A but not B -> FP lost."""
        a = _make_result("q1", ["c1", "c2"], {"c1"})
        b = _make_result("q1", ["c1"], {"c1"})
        rows = compute_retrieval_diff(a, b)

        c2_row = next(r for r in rows if r.chunk_id == "c2")
        assert c2_row.relevant is False
        assert c2_row.status == "FP lost"

    def test_fp_gained(self) -> None:
        """Irrelevant chunk in B but not A -> FP gained."""
        a = _make_result("q1", ["c1"], {"c1"})
        b = _make_result("q1", ["c1", "c2"], {"c1"})
        rows = compute_retrieval_diff(a, b)

        c2_row = next(r for r in rows if r.chunk_id == "c2")
        assert c2_row.relevant is False
        assert c2_row.status == "FP gained"

    def test_moved_up(self) -> None:
        """Chunk present in both, lower rank in B -> Moved up."""
        a = _make_result("q1", ["c1", "c2", "c3"], set())
        b = _make_result("q1", ["c3", "c1", "c2"], set())
        rows = compute_retrieval_diff(a, b)

        c3_row = next(r for r in rows if r.chunk_id == "c3")
        assert c3_row.rank_a == 3
        assert c3_row.rank_b == 1
        assert c3_row.status == "Moved up"

    def test_moved_down(self) -> None:
        """Chunk present in both, higher rank in B -> Moved down."""
        a = _make_result("q1", ["c1", "c2", "c3"], set())
        b = _make_result("q1", ["c3", "c1", "c2"], set())
        rows = compute_retrieval_diff(a, b)

        c1_row = next(r for r in rows if r.chunk_id == "c1")
        assert c1_row.rank_a == 1
        assert c1_row.rank_b == 2
        assert c1_row.status == "Moved down"

    def test_unchanged(self) -> None:
        """Chunk at same rank in both -> Unchanged."""
        a = _make_result("q1", ["c1", "c2"], set())
        b = _make_result("q1", ["c1", "c2"], set())
        rows = compute_retrieval_diff(a, b)

        c1_row = next(r for r in rows if r.chunk_id == "c1")
        assert c1_row.rank_a == 1
        assert c1_row.rank_b == 1
        assert c1_row.status == "Unchanged"

    def test_sort_order_tp_lost_first(self) -> None:
        """TP lost rows sort before FP changes."""
        a = _make_result("q1", ["c1", "c2", "c3"], {"c1"})
        b = _make_result("q1", ["c2", "c4"], {"c1"})
        rows = compute_retrieval_diff(a, b)

        statuses = [r.status for r in rows]
        # TP lost should be first
        assert statuses[0] == "TP lost"

    def test_respects_k(self) -> None:
        """Only considers top-k chunks from each run."""
        a = _make_result("q1", ["c1", "c2", "c3"], set())
        b = _make_result("q1", ["c1", "c2", "c3"], set())
        rows = compute_retrieval_diff(a, b, k=2)

        chunk_ids = {r.chunk_id for r in rows}
        assert "c3" not in chunk_ids  # c3 is at rank 3, beyond k=2

    def test_empty_results(self) -> None:
        """Both runs retrieved nothing -> empty diff table."""
        a = _make_result("q1", [], set())
        b = _make_result("q1", [], set())
        rows = compute_retrieval_diff(a, b)
        assert rows == []

    def test_mismatched_qids_raises(self) -> None:
        """Passing results for different queries raises ValueError."""
        a = _make_result("q1", ["c1"], set())
        b = _make_result("q2", ["c1"], set())

        with pytest.raises(ValueError, match="Cannot diff results for different queries"):
            compute_retrieval_diff(a, b)
