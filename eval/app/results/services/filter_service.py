"""
Service for filtering evaluation results.

Provides flexible filtering by query type, difficulty, pass/fail status,
and other criteria.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.eval.models import EvalResult
from rag.eval.schema import Difficulty, QueryType


def _compute_recall_at_k(result: EvalResult, k: int = 10) -> float:
    """Compute recall@k for a single result."""
    retrieved = set(result.retrieval_result.retrieved_chunk_ids[:k])
    relevant = result.retrieval_result.relevant_chunk_ids

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)


@dataclass
class FilterService:
    """Service for filtering evaluation results."""

    # Default threshold for pass/fail determination
    pass_threshold: float = 0.5

    def filter_results(
        self,
        results: list[EvalResult],
        *,
        query_types: set[QueryType] | None = None,
        difficulties: set[Difficulty] | None = None,
        pass_only: bool = False,
        fail_only: bool = False,
        has_answer: bool | None = None,
        min_recall: float | None = None,
        max_recall: float | None = None,
        min_quality: float | None = None,
        query_search: str | None = None,
    ) -> list[EvalResult]:
        """Apply multiple filters to results.

        Args:
            results: List of results to filter.
            query_types: Only include these query types.
            difficulties: Only include these difficulty levels.
            pass_only: Only include passing queries (recall >= threshold).
            fail_only: Only include failing queries (recall < threshold).
            has_answer: If True, only with answers. If False, only without.
            min_recall: Minimum recall@10 value.
            max_recall: Maximum recall@10 value.
            min_quality: Minimum quality score.
            query_search: Text search in query string (case-insensitive).

        Returns:
            Filtered list of results.
        """
        filtered = list(results)

        if query_types:
            filtered = [
                r for r in filtered
                if r.query_type is not None and r.query_type in query_types
            ]

        if difficulties:
            filtered = [
                r for r in filtered
                if r.difficulty is not None and r.difficulty in difficulties
            ]

        if pass_only:
            filtered = [
                r for r in filtered
                if _compute_recall_at_k(r) >= self.pass_threshold
            ]

        if fail_only:
            filtered = [
                r for r in filtered
                if _compute_recall_at_k(r) < self.pass_threshold
            ]

        if has_answer is not None:
            filtered = [
                r for r in filtered
                if (r.answer is not None) == has_answer
            ]

        if min_recall is not None:
            filtered = [
                r for r in filtered
                if _compute_recall_at_k(r) >= min_recall
            ]

        if max_recall is not None:
            filtered = [
                r for r in filtered
                if _compute_recall_at_k(r) <= max_recall
            ]

        if min_quality is not None:
            filtered = [
                r for r in filtered
                if (
                    r.answer_metrics is not None
                    and r.answer_metrics.quality_score is not None
                    and r.answer_metrics.quality_score >= min_quality
                )
            ]

        if query_search:
            search_lower = query_search.lower()
            filtered = [
                r for r in filtered
                if search_lower in r.query.lower()
            ]

        return filtered

    def compute_recall(self, result: EvalResult, k: int = 10) -> float:
        """Compute recall@k for a single result."""
        return _compute_recall_at_k(result, k)

    def is_passing(self, result: EvalResult) -> bool:
        """Check if a result is passing based on threshold."""
        return _compute_recall_at_k(result) >= self.pass_threshold

    def get_matched_chunks(self, result: EvalResult) -> set[str]:
        """Get chunk IDs that were both retrieved and relevant."""
        retrieved = set(result.retrieval_result.retrieved_chunk_ids)
        relevant = result.retrieval_result.relevant_chunk_ids
        return retrieved & relevant

    def get_missed_chunks(self, result: EvalResult) -> set[str]:
        """Get relevant chunk IDs that were not retrieved."""
        retrieved = set(result.retrieval_result.retrieved_chunk_ids)
        relevant = result.retrieval_result.relevant_chunk_ids
        return relevant - retrieved
