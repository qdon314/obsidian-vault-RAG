"""
Service for comparing two evaluation runs.

Computes metric deltas and identifies improved/regressed queries.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval.app.results.domain.models import LoadedRun, RunComparison
from rag.eval.models import EvalResult


def _compute_recall_at_k(result: EvalResult, k: int = 10) -> float:
    """Compute recall@k for a single result."""
    retrieved = set(result.retrieval_result.retrieved_chunk_ids[:k])
    relevant = result.retrieval_result.relevant_chunk_ids

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)


@dataclass
class ComparisonService:
    """Service for comparing two evaluation runs."""

    # Threshold for considering a query improved/regressed
    change_threshold: float = 0.05

    def compare_runs(self, run_a: LoadedRun, run_b: LoadedRun) -> RunComparison:
        """Generate comprehensive comparison between two runs.

        Args:
            run_a: Baseline run (older or control).
            run_b: Comparison run (newer or experiment).

        Returns:
            RunComparison with all computed deltas and query changes.
        """
        agg_a = run_a.aggregates.overall
        agg_b = run_b.aggregates.overall

        # Compute retrieval metric deltas
        recall_delta = self._compute_metric_delta(
            agg_a.recall_at_k, agg_b.recall_at_k
        )
        precision_delta = self._compute_metric_delta(
            agg_a.precision_at_k, agg_b.precision_at_k
        )
        ndcg_delta = self._compute_metric_delta(
            agg_a.ndcg_at_k, agg_b.ndcg_at_k
        )

        # Global metrics
        mrr_delta = agg_b.mrr - agg_a.mrr
        map_delta = agg_b.map - agg_a.map

        # Answer quality delta
        quality_delta = None
        if run_a.aggregates.answer_quality and run_b.aggregates.answer_quality:
            qa_a = run_a.aggregates.answer_quality.get("avg_quality_score")
            qa_b = run_b.aggregates.answer_quality.get("avg_quality_score")
            if qa_a is not None and qa_b is not None:
                quality_delta = qa_b - qa_a

        # Latency delta
        latency_delta = None
        if run_a.aggregates.latency_ms and run_b.aggregates.latency_ms:
            lat_a = run_a.aggregates.latency_ms.get("avg")
            lat_b = run_b.aggregates.latency_ms.get("avg")
            if lat_a is not None and lat_b is not None:
                latency_delta = lat_b - lat_a

        # Per-query analysis
        improved, regressed, unchanged = self._analyze_query_changes(run_a, run_b)

        return RunComparison(
            run_a=run_a,
            run_b=run_b,
            recall_delta=recall_delta,
            precision_delta=precision_delta,
            ndcg_delta=ndcg_delta,
            mrr_delta=mrr_delta,
            map_delta=map_delta,
            quality_delta=quality_delta,
            latency_delta=latency_delta,
            improved_queries=tuple(improved),
            regressed_queries=tuple(regressed),
            unchanged_queries=tuple(unchanged),
        )

    def _compute_metric_delta(
        self, a: dict[int, float], b: dict[int, float]
    ) -> dict[int, float]:
        """Compute b - a for each k value."""
        all_keys = set(a.keys()) | set(b.keys())
        return {k: b.get(k, 0.0) - a.get(k, 0.0) for k in all_keys}

    def _analyze_query_changes(
        self, run_a: LoadedRun, run_b: LoadedRun
    ) -> tuple[list[str], list[str], list[str]]:
        """Identify improved, regressed, and unchanged queries.

        Returns:
            Tuple of (improved_qids, regressed_qids, unchanged_qids).
        """
        improved: list[str] = []
        regressed: list[str] = []
        unchanged: list[str] = []

        # Build qid -> result maps
        a_results = {r.qid: r for r in run_a.results}
        b_results = {r.qid: r for r in run_b.results}

        # Only compare queries present in both runs
        common_qids = set(a_results.keys()) & set(b_results.keys())

        for qid in common_qids:
            a_recall = _compute_recall_at_k(a_results[qid], k=10)
            b_recall = _compute_recall_at_k(b_results[qid], k=10)

            delta = b_recall - a_recall

            if delta > self.change_threshold:
                improved.append(qid)
            elif delta < -self.change_threshold:
                regressed.append(qid)
            else:
                unchanged.append(qid)

        return improved, regressed, unchanged
