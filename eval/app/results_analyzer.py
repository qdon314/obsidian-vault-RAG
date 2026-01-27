#!/usr/bin/env python3
"""
Evaluation Results Analyzer - Streamlit App

A comprehensive tool for analyzing, comparing, and trending RAG evaluation runs.

Features:
- Single run analysis with detailed metrics and query explorer
- Side-by-side comparison of two runs with delta highlighting
- Multi-run trending with time series charts
- Flexible filtering and drill-down capabilities

Usage:
    streamlit run eval/app/results_analyzer.py

    Or via Makefile:
    make results
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from eval.app.results.adapters.filesystem_loader import FilesystemRunLoader
from eval.app.results.adapters.repository import InMemoryRunRepository
from eval.app.results.domain.models import LoadedRun, RunSummary
from eval.app.results.services.comparison_service import ComparisonService
from eval.app.results.services.filter_service import FilterService
from eval.app.results.services.trend_service import TrendService
from eval.app.results.ui.comparison_chart import (
    render_comparison_chart,
    render_global_metrics_comparison,
)
from eval.app.results.ui.delta_table import (
    render_delta_table,
    render_query_changes_summary,
    render_summary_metrics,
)
from eval.app.results.ui.metrics_table import render_metrics_table
from eval.app.results.ui.query_explorer import render_query_explorer
from eval.app.results.ui.run_selector import render_run_info_card, render_run_selector
from eval.app.results.ui.trend_chart import render_multi_metric_trend, render_trend_chart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine project paths
PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "eval" / "runs"


# Initialize services as cached resources
@st.cache_resource
def get_repository() -> InMemoryRunRepository:
    """Initialize the run repository."""
    runs_dir = DEFAULT_RUNS_DIR
    if not runs_dir.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)

    loader = FilesystemRunLoader(runs_dir=runs_dir)
    return InMemoryRunRepository(loader=loader)


@st.cache_resource
def get_services() -> dict:
    """Initialize business logic services."""
    return {
        "comparison": ComparisonService(),
        "filter": FilterService(),
        "trend": TrendService(),
    }


def get_theme_css() -> str:
    """Generate minimal CSS enhancements that work with any theme.

    We rely on Streamlit's native theming and only add subtle enhancements.
    """
    return """
    <style>
        .stMetric {
            padding: 10px;
            border-radius: 5px;
        }
    </style>
    """


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="RAG Eval Results Analyzer",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize theme in session state
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    # Apply minimal theme-agnostic CSS
    st.markdown(get_theme_css(), unsafe_allow_html=True)

    repo = get_repository()
    services = get_services()

    # Sidebar navigation
    with st.sidebar:
        st.title("Results Analyzer")
        
        view = st.radio(
            "View Mode",
            options=["single", "comparison", "trending"],
            format_func=lambda v: {
                "single": "Single Run Analysis",
                "comparison": "Compare Two Runs",
                "trending": "Multi-Run Trending",
            }[v],
            key="view_mode",
        )

        st.divider()

        # Manual path addition
        st.subheader("Add External Run")
        external_path = st.text_input(
            "Run directory path",
            placeholder="/path/to/run_directory",
            key="external_path",
        )
        if st.button("Add Run", key="add_run_btn") and external_path:
            try:
                path = Path(external_path).expanduser().resolve()
                summary = repo.add_run_path(path)
                st.success(f"Added: {summary.display_name}")
                st.rerun()
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Failed to add run: {e}")

        st.divider()

        # Refresh button
        if st.button("Refresh Runs", key="refresh_btn"):
            repo.refresh()
            st.cache_resource.clear()
            st.rerun()

        st.divider()

        # Info
        st.caption(f"Runs directory: {DEFAULT_RUNS_DIR}")

    # Discover available runs
    try:
        available_runs = repo.list_runs()
    except Exception as e:
        st.error(f"Failed to load runs: {e}")
        logger.exception("Failed to load runs")
        return

    if not available_runs:
        st.warning("No evaluation runs found")
        st.info(
            "Run `make eval` to create evaluation runs, "
            "or add external runs using the sidebar."
        )
        st.caption(f"Looking in: {DEFAULT_RUNS_DIR}")
        return

    # Main content based on view mode
    if view == "single":
        render_single_run_view(repo, services, available_runs)
    elif view == "comparison":
        render_comparison_view(repo, services, available_runs)
    elif view == "trending":
        render_trending_view(repo, services, available_runs)


def render_single_run_view(
    repo: InMemoryRunRepository,
    services: dict,
    available_runs: list[RunSummary],
) -> None:
    """Render single run analysis view."""
    st.header("Single Run Analysis")

    # Run selection
    selected_id = render_run_selector(
        available_runs,
        key="single_run_selector",
        multi=False,
        label="Select a run to analyze",
    )

    if not selected_id:
        st.info("Select a run from the dropdown above to view its analysis")
        return

    # Load the run
    try:
        with st.spinner("Loading run data..."):
            loaded_run = repo.get_run(selected_id)
    except Exception as e:
        st.error(f"Failed to load run: {e}")
        logger.exception(f"Failed to load run {selected_id}")
        return

    # Run info card
    with st.expander("Run Configuration", expanded=False):
        render_run_info_card(loaded_run.summary)

        # Additional meta info
        st.markdown("**Run Metadata**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"- **Run ID:** `{loaded_run.meta.run_id}`")
            st.markdown(f"- **Top K:** {loaded_run.meta.top_k}")
            st.markdown(f"- **Token Budget:** {loaded_run.meta.token_budget}")
        with col2:
            st.markdown(f"- **Generation:** {'Yes' if loaded_run.meta.run_generation else 'No'}")
            st.markdown(f"- **LLM Judge:** {'Yes' if loaded_run.meta.use_llm_judge else 'No'}")
            if loaded_run.meta.notes:
                st.markdown(f"- **Notes:** {loaded_run.meta.notes}")

    # Tabs for different views
    tab_metrics, tab_charts, tab_explorer, tab_traces, tab_raw = st.tabs([
        "Metrics",
        "Charts",
        "Query Explorer",
        "Traces",
        "Raw Data",
    ])

    with tab_metrics:
        render_metrics_table(
            loaded_run,
            show_by_type=True,
            show_by_difficulty=True,
        )

    with tab_charts:
        _render_single_run_charts(loaded_run)

    with tab_explorer:
        render_query_explorer(loaded_run, services["filter"])

    with tab_traces:
        _render_traces_tab(loaded_run)

    with tab_raw:
        _render_raw_data_tab(loaded_run)


def _render_single_run_charts(loaded_run: LoadedRun) -> None:
    """Render charts for a single run."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("Plotly is required for charts. Install with: pip install plotly")
        return

    st.subheader("Retrieval Metrics")

    # Bar chart of metrics at different k values
    agg = loaded_run.aggregates.overall
    k_values = sorted(agg.recall_at_k.keys())

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[f"@{k}" for k in k_values],
        y=[agg.recall_at_k.get(k, 0) for k in k_values],
        name="Recall",
    ))

    fig.add_trace(go.Bar(
        x=[f"@{k}" for k in k_values],
        y=[agg.precision_at_k.get(k, 0) for k in k_values],
        name="Precision",
    ))

    fig.add_trace(go.Bar(
        x=[f"@{k}" for k in k_values],
        y=[agg.ndcg_at_k.get(k, 0) for k in k_values],
        name="NDCG",
    ))

    fig.update_layout(
        title="Retrieval Metrics by K",
        xaxis_title="K",
        yaxis_title="Score",
        barmode="group",
        height=400,
        yaxis={"range": [0, 1.05]},
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics by query type
    if loaded_run.aggregates.by_type:
        st.subheader("Recall@10 by Query Type")

        types = sorted(loaded_run.aggregates.by_type.keys())
        recalls = [
            loaded_run.aggregates.by_type[t].recall_at_k.get(10, 0)
            for t in types
        ]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=types,
            y=recalls,
            text=[f"{r:.2f}" for r in recalls],
            textposition="auto",
        ))

        fig2.update_layout(
            title="Recall@10 by Query Type",
            xaxis_title="Query Type",
            yaxis_title="Recall@10",
            height=350,
            yaxis={"range": [0, 1.05]},
        )

        st.plotly_chart(fig2, use_container_width=True)


def _render_traces_tab(loaded_run: LoadedRun) -> None:
    """Render the traces tab for a single run."""
    if not loaded_run.traces:
        st.info("No traces available for this run")
        st.caption("Traces are recorded in traces.jsonl during evaluation runs with generation enabled.")
        return

    st.subheader(f"Pipeline Traces ({len(loaded_run.traces)} total)")

    # Allow searching/filtering traces
    search = st.text_input("Search traces by query text", key="trace_search")

    # Filter traces
    traces_list = list(loaded_run.traces.values())
    if search:
        search_lower = search.lower()
        traces_list = [t for t in traces_list if search_lower in t.query.lower()]

    st.write(f"Showing {len(traces_list)} traces")

    if not traces_list:
        st.info("No traces match the search")
        return

    # Select trace to view
    selected_trace_id = st.selectbox(
        "Select trace to view",
        options=[t.trace_id for t in traces_list],
        format_func=lambda tid: next(
            (f"{tid[:8]}... | {t.query[:50]}..." for t in traces_list if t.trace_id == tid),
            tid
        ),
        key="trace_tab_selector",
    )

    if selected_trace_id:
        trace = loaded_run.traces[selected_trace_id]
        _render_full_trace(trace)


def _render_full_trace(trace) -> None:
    """Render a complete trace with all pipeline stages."""
    st.markdown(f"**Trace ID:** `{trace.trace_id}`")
    st.markdown(f"**Query:** {trace.query}")

    if trace.created_at:
        st.caption(f"Created: {trace.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top K", trace.top_k)
    with col2:
        st.metric("Retrieved", len(trace.retrieved_candidates))
    with col3:
        st.metric("Latency", f"{trace.latency_ms} ms" if trace.latency_ms else "N/A")

    st.divider()

    # Retrieval stage
    with st.expander("Retrieval Stage", expanded=True):
        if trace.retrieved_candidates:
            for i, cand in enumerate(trace.retrieved_candidates, 1):
                chunk = cand.get("chunk", {})
                score = cand.get("score", 0)
                chunk_id = chunk.get("chunk_id", "unknown")
                text = chunk.get("text", "")
                section = chunk.get("section_heading", "")

                st.markdown(f"**{i}. `{chunk_id}`** (score: {score:.4f})")
                if section:
                    st.caption(f"Section: {section}")
                st.text_area(
                    f"Text (rank {i})",
                    value=text,
                    height=100,
                    disabled=True,
                    key=f"trace_chunk_{trace.trace_id}_{i}",
                )
        else:
            st.info("No retrieval data")

    # Reranking stage
    if trace.reranked_candidates:
        with st.expander("Reranking Stage", expanded=False):
            st.markdown(f"**Reranker:** {trace.reranker or 'N/A'}")
            st.markdown(f"**Keep K:** {trace.keep_k or 'All'}")

            for i, cand in enumerate(trace.reranked_candidates, 1):
                chunk = cand.get("chunk", {})
                score = cand.get("score", 0)
                rerank_score = cand.get("rerank_score")
                chunk_id = chunk.get("chunk_id", "unknown")

                score_info = f"orig: {score:.4f}"
                if rerank_score is not None:
                    score_info += f", rerank: {rerank_score:.4f}"

                st.markdown(f"**{i}. `{chunk_id}`** ({score_info})")

    # Context building stage
    if trace.packed_chunk_ids:
        with st.expander("Context Building Stage", expanded=False):
            st.markdown(f"**Token Budget:** {trace.token_budget}")
            st.markdown(f"**Chunks Packed:** {len(trace.packed_chunk_ids)}")

            for i, chunk_id in enumerate(trace.packed_chunk_ids, 1):
                st.code(f"{i}. {chunk_id}", language=None)

    # Generation stage
    with st.expander("Generation Stage", expanded=False):
        st.markdown(f"**Model:** {trace.model or 'N/A'}")

        if trace.answer_text:
            st.text_area(
                "Generated Answer",
                value=trace.answer_text,
                height=200,
                disabled=True,
                key=f"trace_answer_{trace.trace_id}",
            )

            if trace.citations:
                st.markdown(f"**Citations ({len(trace.citations)}):**")
                for i, cit in enumerate(trace.citations, 1):
                    chunk_id = cit.get("chunk_id", "unknown")
                    quote = cit.get("quote", "")
                    st.markdown(f"**[{i}]** `{chunk_id}`")
                    if quote:
                        st.caption(f'"{quote[:150]}..."' if len(quote) > 150 else f'"{quote}"')
        else:
            st.info("No answer generated")

    # Raw data
    with st.expander("Raw Trace Data", expanded=False):
        st.json(trace.raw_data)


def _render_raw_data_tab(loaded_run: LoadedRun) -> None:
    """Render the raw data tab showing metrics.json contents."""
    st.subheader("Raw Metrics Data")

    if not loaded_run.raw_metrics:
        st.info("No raw metrics data available")
        return

    # Show sections as expandable
    raw = loaded_run.raw_metrics

    # Meta section
    if "meta" in raw:
        with st.expander("Run Metadata (meta)", expanded=True):
            st.json(raw["meta"])

    # Overall metrics
    if "overall" in raw:
        with st.expander("Overall Metrics", expanded=True):
            st.json(raw["overall"])

    # By type
    if "by_type" in raw:
        with st.expander("Metrics by Query Type", expanded=False):
            st.json(raw["by_type"])

    # By difficulty
    if "by_difficulty" in raw:
        with st.expander("Metrics by Difficulty", expanded=False):
            st.json(raw["by_difficulty"])

    # Answer quality
    if "answer_quality" in raw:
        with st.expander("Answer Quality Metrics", expanded=False):
            st.json(raw["answer_quality"])

    # Latency
    if "latency_ms" in raw:
        with st.expander("Latency Statistics", expanded=False):
            st.json(raw["latency_ms"])

    # Full JSON download
    st.divider()
    st.markdown("### Download Raw Data")

    import json
    json_str = json.dumps(raw, indent=2, default=str)
    st.download_button(
        label="Download metrics.json",
        data=json_str,
        file_name=f"{loaded_run.summary.display_name}_metrics.json",
        mime="application/json",
    )


def render_comparison_view(
    repo: InMemoryRunRepository,
    services: dict,
    available_runs: list[RunSummary],
) -> None:
    """Render two-run comparison view."""
    st.header("Compare Two Runs")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Run A (Baseline)")
        run_a_id = render_run_selector(
            available_runs,
            key="compare_run_a",
            multi=False,
            label="Select baseline run",
        )

    with col2:
        st.subheader("Run B (Comparison)")
        run_b_id = render_run_selector(
            available_runs,
            key="compare_run_b",
            multi=False,
            label="Select comparison run",
        )

    if not (run_a_id and run_b_id):
        st.info("Select two runs to compare them")
        return

    if run_a_id == run_b_id:
        st.warning("Please select different runs for comparison")
        return

    # Load runs
    try:
        with st.spinner("Loading runs..."):
            run_a = repo.get_run(run_a_id)
            run_b = repo.get_run(run_b_id)
    except Exception as e:
        st.error(f"Failed to load runs: {e}")
        return

    # Compute comparison
    with st.spinner("Computing comparison..."):
        comparison = services["comparison"].compare_runs(run_a, run_b)

    st.divider()

    # Summary metrics with deltas
    st.subheader("Summary")
    render_summary_metrics(comparison)

    st.divider()

    # Query-level changes summary
    st.subheader("Query-Level Changes")
    render_query_changes_summary(comparison)

    # Tabs for detailed views
    tab_charts, tab_deltas, tab_queries = st.tabs([
        "Charts",
        "Delta Table",
        "Query Changes",
    ])

    with tab_charts:
        col1, col2 = st.columns(2)
        with col1:
            render_comparison_chart(comparison, metric="recall")
        with col2:
            render_comparison_chart(comparison, metric="ndcg")

        render_global_metrics_comparison(comparison)

    with tab_deltas:
        render_delta_table(comparison)

    with tab_queries:
        _render_query_changes(comparison, services["filter"])


def _render_query_changes(comparison, filter_service: FilterService) -> None:
    """Render detailed query-level changes."""
    if comparison.improved_queries:
        st.subheader(f"Improved Queries ({len(comparison.improved_queries)})")
        with st.expander("Show improved queries", expanded=True):
            for qid in comparison.improved_queries[:20]:  # Limit display
                result = next(
                    (r for r in comparison.run_b.results if r.qid == qid),
                    None
                )
                if result:
                    recall_a = _get_recall(comparison.run_a, qid)
                    recall_b = filter_service.compute_recall(result)
                    st.success(
                        f"**{qid}**: {result.query[:80]}... "
                        f"(Recall: {recall_a:.2f} -> {recall_b:.2f})"
                    )
            if len(comparison.improved_queries) > 20:
                st.caption(f"...and {len(comparison.improved_queries) - 20} more")
    else:
        st.info("No queries improved significantly")

    if comparison.regressed_queries:
        st.subheader(f"Regressed Queries ({len(comparison.regressed_queries)})")
        with st.expander("Show regressed queries", expanded=True):
            for qid in comparison.regressed_queries[:20]:
                result = next(
                    (r for r in comparison.run_b.results if r.qid == qid),
                    None
                )
                if result:
                    recall_a = _get_recall(comparison.run_a, qid)
                    recall_b = filter_service.compute_recall(result)
                    st.error(
                        f"**{qid}**: {result.query[:80]}... "
                        f"(Recall: {recall_a:.2f} -> {recall_b:.2f})"
                    )
            if len(comparison.regressed_queries) > 20:
                st.caption(f"...and {len(comparison.regressed_queries) - 20} more")
    else:
        st.info("No queries regressed significantly")


def _get_recall(run: LoadedRun, qid: str) -> float:
    """Get recall@10 for a specific query in a run."""
    result = next((r for r in run.results if r.qid == qid), None)
    if not result:
        return 0.0

    retrieved = set(result.retrieval_result.retrieved_chunk_ids[:10])
    relevant = result.retrieval_result.relevant_chunk_ids
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)


def render_trending_view(
    repo: InMemoryRunRepository,
    services: dict,
    available_runs: list[RunSummary],
) -> None:
    """Render multi-run trending view."""
    st.header("Multi-Run Trending")

    # Multi-select for runs
    selected_ids = render_run_selector(
        available_runs,
        key="trend_runs",
        multi=True,
        label="Select runs for trend analysis (minimum 2)",
    )

    if not selected_ids or len(selected_ids) < 2:
        st.info("Select at least 2 runs to see trends over time")
        return

    # Load runs
    try:
        with st.spinner(f"Loading {len(selected_ids)} runs..."):
            loaded_runs = [repo.get_run(rid) for rid in selected_ids]
    except Exception as e:
        st.error(f"Failed to load runs: {e}")
        return

    # Compute trend analysis
    with st.spinner("Analyzing trends..."):
        trend = services["trend"].analyze_trends(loaded_runs)

    date_range = trend.date_range
    if date_range:
        st.write(f"Analyzing {trend.num_runs} runs from {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")
    else:
        st.write(f"Analyzing {trend.num_runs} runs")

    st.divider()

    # Metric selector
    col1, col2 = st.columns([1, 3])

    with col1:
        metric_choice = st.selectbox(
            "Metric",
            options=["recall", "precision", "ndcg", "mrr", "map", "quality", "latency"],
            format_func=lambda m: {
                "recall": "Recall",
                "precision": "Precision",
                "ndcg": "NDCG",
                "mrr": "MRR",
                "map": "MAP",
                "quality": "Quality Score",
                "latency": "Latency",
            }[m],
            key="trend_metric",
        )

        if metric_choice in ["recall", "precision", "ndcg"]:
            k_choice = st.selectbox(
                "K value",
                options=[1, 3, 5, 10],
                index=3,  # Default to 10
                key="trend_k",
            )
        else:
            k_choice = 10

    with col2:
        if metric_choice in ["recall", "precision", "ndcg"]:
            render_trend_chart(trend, metric=metric_choice, k=k_choice)
        else:
            render_trend_chart(trend, metric=metric_choice)

    st.divider()

    # Multi-metric comparison
    st.subheader("Multi-Metric Trend")
    render_multi_metric_trend(trend, metrics=["recall", "precision", "ndcg"], k=10)

    st.divider()

    # Summary table
    st.subheader("Run Summary Table")
    _render_trend_summary_table(trend)


def _render_trend_summary_table(trend) -> None:
    """Render summary table for trending analysis."""
    import pandas as pd

    data = []
    for run in trend.runs:
        row = {
            "Timestamp": run.summary.timestamp.strftime("%Y-%m-%d %H:%M"),
            "Run": run.summary.display_name,
            "Recall@10": f"{run.aggregates.overall.recall_at_k.get(10, 0):.3f}",
            "NDCG@10": f"{run.aggregates.overall.ndcg_at_k.get(10, 0):.3f}",
            "MRR": f"{run.aggregates.overall.mrr:.3f}",
            "Queries": run.summary.num_queries,
        }

        if run.aggregates.answer_quality:
            row["Quality"] = f"{run.aggregates.answer_quality.get('avg_quality_score', 0):.3f}"
        else:
            row["Quality"] = "N/A"

        if run.aggregates.latency_ms:
            row["Latency (ms)"] = f"{run.aggregates.latency_ms.get('avg', 0):.0f}"
        else:
            row["Latency (ms)"] = "N/A"

        data.append(row)

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
