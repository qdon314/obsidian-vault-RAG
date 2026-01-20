"""
Service for multi-run trend analysis.

Builds time series data from multiple evaluation runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval.app.results.domain.models import LoadedRun, TrendAnalysis


@dataclass
class TrendService:
    """Service for analyzing trends across multiple evaluation runs."""

    def analyze_trends(self, runs: list[LoadedRun]) -> TrendAnalysis:
        """Build trend analysis from multiple runs.

        Args:
            runs: List of loaded runs to analyze. Will be sorted by timestamp.

        Returns:
            TrendAnalysis with time series for all key metrics.

        Raises:
            ValueError: If fewer than 2 runs provided.
        """
        if len(runs) < 2:
            raise ValueError("Trend analysis requires at least 2 runs")

        # Validate all runs have timestamps
        if any(r.summary.timestamp is None for r in runs):
            raise ValueError("All runs must have valid timestamps for trend analysis")

        # Sort by timestamp
        sorted_runs = sorted(runs, key=lambda r: r.summary.timestamp)

        timestamps = tuple(r.summary.timestamp for r in sorted_runs)

        # Build retrieval metric series
        k_values = [1, 3, 5, 10]

        recall_series: dict[int, tuple[float, ...]] = {}
        precision_series: dict[int, tuple[float, ...]] = {}
        ndcg_series: dict[int, tuple[float, ...]] = {}

        for k in k_values:
            recall_series[k] = tuple(
                r.aggregates.overall.recall_at_k.get(k, 0.0)
                for r in sorted_runs
            )
            precision_series[k] = tuple(
                r.aggregates.overall.precision_at_k.get(k, 0.0)
                for r in sorted_runs
            )
            ndcg_series[k] = tuple(
                r.aggregates.overall.ndcg_at_k.get(k, 0.0)
                for r in sorted_runs
            )

        # Global metrics series
        mrr_series = tuple(r.aggregates.overall.mrr for r in sorted_runs)
        map_series = tuple(r.aggregates.overall.map for r in sorted_runs)

        # Answer quality series
        quality_series: list[float | None] = []
        for r in sorted_runs:
            if r.aggregates.answer_quality:
                quality_series.append(
                    r.aggregates.answer_quality.get("avg_quality_score")
                )
            else:
                quality_series.append(None)

        # Latency series
        latency_series: list[float | None] = []
        for r in sorted_runs:
            if r.aggregates.latency_ms:
                latency_series.append(r.aggregates.latency_ms.get("avg"))
            else:
                latency_series.append(None)

        return TrendAnalysis(
            runs=tuple(sorted_runs),
            timestamps=timestamps,
            recall_series=recall_series,
            precision_series=precision_series,
            ndcg_series=ndcg_series,
            mrr_series=mrr_series,
            map_series=map_series,
            quality_series=tuple(quality_series),
            latency_series=tuple(latency_series),
        )

    def compute_improvement(
        self, trend: TrendAnalysis, metric: str = "recall", k: int = 10
    ) -> float:
        """Compute overall improvement from first to last run.

        Args:
            trend: TrendAnalysis to analyze.
            metric: Metric name ("recall", "precision", "ndcg", "mrr", "map").
            k: K value for @k metrics.

        Returns:
            Difference between last and first run (positive = improvement).
        """
        if metric == "recall":
            series = trend.recall_series.get(k, ())
        elif metric == "precision":
            series = trend.precision_series.get(k, ())
        elif metric == "ndcg":
            series = trend.ndcg_series.get(k, ())
        elif metric == "mrr":
            series = trend.mrr_series
        elif metric == "map":
            series = trend.map_series
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if len(series) < 2:
            return 0.0

        return series[-1] - series[0]

    def find_best_run(
        self, trend: TrendAnalysis, metric: str = "recall", k: int = 10
    ) -> LoadedRun:
        """Find the run with the best value for a given metric.

        Args:
            trend: TrendAnalysis to search.
            metric: Metric name to optimize.
            k: K value for @k metrics.

        Returns:
            LoadedRun with the highest metric value.
        """
        if metric == "recall":
            series = trend.recall_series.get(k, ())
        elif metric == "precision":
            series = trend.precision_series.get(k, ())
        elif metric == "ndcg":
            series = trend.ndcg_series.get(k, ())
        elif metric == "mrr":
            series = trend.mrr_series
        elif metric == "map":
            series = trend.map_series
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if not series:
            return trend.runs[0]

        best_idx = max(range(len(series)), key=lambda i: series[i])
        return trend.runs[best_idx]
