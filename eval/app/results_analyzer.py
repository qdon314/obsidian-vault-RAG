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
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from eval.app.results.adapters.filesystem_loader import FilesystemRunLoader
from eval.app.results.adapters.repository import InMemoryRunRepository
from eval.app.results.adapters.s3_loader import S3RunLoader
from eval.app.results.domain.models import LoadedRun, RunSummary
from eval.app.results.query_diff import compute_retrieval_diff, natural_sort_key
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
from rag.eval.models import EvalResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine project paths
PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "eval" / "runs"


# Initialize services as cached resources
@st.cache_resource
def get_repository() -> InMemoryRunRepository:
    """Initialize the run repository.

    If RAG_EVAL_S3_BUCKET is set (or settings.toml has an S3 bucket),
    an S3RunLoader is wired in alongside the filesystem loader. Runs
    from both sources appear in a unified list.
    """
    runs_dir = DEFAULT_RUNS_DIR
    if not runs_dir.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)

    loader = FilesystemRunLoader(runs_dir=runs_dir)

    # Conditionally wire S3 loader
    s3_loader: S3RunLoader | None = None
    s3_bucket = os.environ.get("RAG_EVAL_S3_BUCKET", "")
    s3_prefix = os.environ.get("RAG_EVAL_S3_PREFIX", "eval")

    if s3_bucket:
        try:
            s3_cache_dir = runs_dir / ".s3-cache"
            s3_loader = S3RunLoader(
                bucket=s3_bucket,
                prefix=f"{s3_prefix}/runs",
                cache_dir=s3_cache_dir,
            )
            logger.info("S3 run loader enabled: s3://%s/%s/runs", s3_bucket, s3_prefix)
        except Exception:
            logger.warning("Failed to initialize S3 run loader", exc_info=True)

    return InMemoryRunRepository(loader=loader, s3_loader=s3_loader)


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
        s3_bucket = os.environ.get("RAG_EVAL_S3_BUCKET", "")
        if s3_bucket:
            s3_prefix = os.environ.get("RAG_EVAL_S3_PREFIX", "eval")
            st.caption(f"S3: s3://{s3_bucket}/{s3_prefix}/runs/")

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
            "Run `make eval` to create evaluation runs, or add external runs using the sidebar."
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"- **Run ID:** `{loaded_run.meta.run_id}`")
            st.markdown(f"- **Started At:** `{loaded_run.meta.started_at.isoformat()}`")
            st.markdown(f"- **Run Name:** {loaded_run.meta.run_name or 'n/a'}")
            st.markdown(f"- **Queries File:** `{loaded_run.meta.queries_path or 'n/a'}`")
            st.markdown(f"- **Index Name:** `{loaded_run.meta.index_name or 'n/a'}`")
            st.markdown(f"- **Index ID:** `{loaded_run.meta.index_id or 'n/a'}`")
            st.markdown(f"- **Index Build ID:** `{loaded_run.meta.index_build_id or 'n/a'}`")
        with col2:
            st.markdown(f"- **Top K:** {loaded_run.meta.top_k}")
            st.markdown(
                f"- **Keep K:** {loaded_run.meta.keep_k if loaded_run.meta.keep_k is not None else 'n/a'}"
            )
            st.markdown(f"- **Token Budget:** {loaded_run.meta.token_budget}")
            retrieval_backend = loaded_run.meta.extra.get("retrieval_backend", "n/a")
            st.markdown(f"- **Retrieval Backend:** `{retrieval_backend}`")
            st.markdown(f"- **Score IDs:** `{loaded_run.meta.extra.get('score_ids', 'n/a')}`")
            if retrieval_backend == "hybrid":
                st.markdown(
                    "- **Hybrid Knobs:** "
                    f"`primary={loaded_run.meta.extra.get('hybrid_primary_weight', 'n/a')}`, "
                    f"`secondary={loaded_run.meta.extra.get('hybrid_secondary_weight', 'n/a')}`, "
                    f"`rrf_k={loaded_run.meta.extra.get('hybrid_rrf_k', 'n/a')}`, "
                    f"`bm25_k1={loaded_run.meta.extra.get('hybrid_bm25_k1', 'n/a')}`, "
                    f"`bm25_b={loaded_run.meta.extra.get('hybrid_bm25_b', 'n/a')}`"
                )
            st.markdown(f"- **Generation:** {'Yes' if loaded_run.meta.run_generation else 'No'}")
            st.markdown(f"- **LLM Judge:** {'Yes' if loaded_run.meta.use_llm_judge else 'No'}")
            st.markdown(f"- **Judge Model:** `{loaded_run.meta.judge_model or 'n/a'}`")
            st.markdown(
                f"- **Gold Judge Version:** `{loaded_run.meta.gold_judge_version or 'n/a'}`"
            )
            st.markdown(
                f"- **Groundedness Judge Version:** "
                f"`{loaded_run.meta.groundedness_judge_version or 'n/a'}`"
            )
        with col3:
            st.markdown(f"- **Generator Model:** `{loaded_run.meta.generator_model or 'n/a'}`")
            st.markdown(f"- **Embedder Model:** `{loaded_run.meta.embedder_model or 'n/a'}`")
            st.markdown(f"- **Reranker:** `{loaded_run.meta.reranker_name or 'n/a'}`")
            if loaded_run.meta.notes:
                st.markdown(f"- **Notes:** {loaded_run.meta.notes}")

        if loaded_run.meta.extra:
            with st.expander("Additional Metadata (meta.extra)", expanded=False):
                st.json(loaded_run.meta.extra)

    # Initialize tab selection in session state if not present
    if "single_run_tab" not in st.session_state:
        st.session_state.single_run_tab = "Metrics"

    # Tab selection using radio for persistence
    tab_names = ["Metrics", "Charts", "Query Explorer", "Traces", "Raw Data"]
    selected_tab = st.radio(
        "View",
        tab_names,
        index=tab_names.index(st.session_state.single_run_tab),
        horizontal=True,
        key="single_run_tab_radio",
        label_visibility="collapsed",
    )
    st.session_state.single_run_tab = selected_tab

    # Render content based on selected tab
    if selected_tab == "Metrics":
        render_metrics_table(
            loaded_run,
            show_by_type=True,
            show_by_difficulty=True,
        )
    elif selected_tab == "Charts":
        _render_single_run_charts(loaded_run)
    elif selected_tab == "Query Explorer":
        render_query_explorer(loaded_run, services["filter"])
    elif selected_tab == "Traces":
        _render_traces_tab(loaded_run)
    elif selected_tab == "Raw Data":
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

    fig.add_trace(
        go.Bar(
            x=[f"@{k}" for k in k_values],
            y=[agg.recall_at_k.get(k, 0) for k in k_values],
            name="Recall",
        )
    )

    fig.add_trace(
        go.Bar(
            x=[f"@{k}" for k in k_values],
            y=[agg.precision_at_k.get(k, 0) for k in k_values],
            name="Precision",
        )
    )

    fig.add_trace(
        go.Bar(
            x=[f"@{k}" for k in k_values],
            y=[agg.ndcg_at_k.get(k, 0) for k in k_values],
            name="NDCG",
        )
    )
    if agg.critical_recall_at_k:
        fig.add_trace(
            go.Bar(
                x=[f"@{k}" for k in k_values],
                y=[agg.critical_recall_at_k.get(k, 0) for k in k_values],
                name="Critical Recall",
            )
        )
    if agg.weighted_recall_at_k:
        fig.add_trace(
            go.Bar(
                x=[f"@{k}" for k in k_values],
                y=[agg.weighted_recall_at_k.get(k, 0) for k in k_values],
                name="Weighted Recall",
            )
        )
    if agg.critical_hit_rate_at_k:
        fig.add_trace(
            go.Bar(
                x=[f"@{k}" for k in k_values],
                y=[agg.critical_hit_rate_at_k.get(k, 0) for k in k_values],
                name="Critical Hit Rate",
            )
        )

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
        recalls = [loaded_run.aggregates.by_type[t].recall_at_k.get(10, 0) for t in types]

        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=types,
                y=recalls,
                text=[f"{r:.2f}" for r in recalls],
                textposition="auto",
            )
        )

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
        st.caption(
            "Traces are recorded in traces.jsonl during evaluation runs with generation enabled."
        )
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
            (f"{tid[:8]}... | {t.query[:50]}..." for t in traces_list if t.trace_id == tid), tid
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
    with st.expander("Retrieval Stage", expanded=False):
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

    # Compression
    if raw.get("compression"):
        with st.expander("Compression Metrics (ScaleDown)", expanded=False):
            st.json(raw["compression"])

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
    tab_charts, tab_deltas, tab_queries = st.tabs(
        [
            "Charts",
            "Delta Table",
            "Query Changes",
        ]
    )

    with tab_charts:
        col1, col2 = st.columns(2)
        with col1:
            render_comparison_chart(comparison, metric="recall")
        with col2:
            render_comparison_chart(comparison, metric="ndcg")

        if (
            comparison.run_a.aggregates.overall.critical_recall_at_k
            or comparison.run_b.aggregates.overall.critical_recall_at_k
        ):
            col3, col4 = st.columns(2)
            with col3:
                render_comparison_chart(comparison, metric="critical_recall")
            with col4:
                render_comparison_chart(comparison, metric="weighted_recall")

        render_global_metrics_comparison(comparison)

    with tab_deltas:
        render_delta_table(comparison)

    with tab_queries:
        _render_query_changes(comparison)


def _render_query_changes(comparison) -> None:
    """Render detailed query-level changes with drill-down diagnostics."""
    # Build lookup dicts once to avoid O(n) scans per query
    results_a: dict[str, EvalResult] = {r.qid: r for r in comparison.run_a.results}
    results_b: dict[str, EvalResult] = {r.qid: r for r in comparison.run_b.results}

    sorted_improved = sorted(comparison.improved_queries, key=natural_sort_key)
    sorted_regressed = sorted(comparison.regressed_queries, key=natural_sort_key)

    _render_query_category(
        "Improved",
        sorted_improved,
        results_a,
        results_b,
    )
    _render_query_category(
        "Regressed",
        sorted_regressed,
        results_a,
        results_b,
    )


def _render_query_category(
    label: str,
    qids: list[str],
    results_a: dict[str, EvalResult],
    results_b: dict[str, EvalResult],
) -> None:
    """Render a category (improved/regressed) of changed queries."""
    if not qids:
        st.info(f"No queries {label.lower()} significantly")
        return

    st.subheader(f"{label} Queries ({len(qids)})")

    for qid in qids[:20]:
        result_a = results_a.get(qid)
        result_b = results_b.get(qid)
        if not result_a or not result_b:
            continue

        recall_a = _compute_recall(result_a)
        recall_b = _compute_recall(result_b)

        # Summary line for expander label
        query_type = result_b.query_type.value if result_b.query_type else "—"
        difficulty = result_b.difficulty.value if result_b.difficulty else "—"
        delta = recall_b - recall_a
        sign = "+" if delta >= 0 else ""
        summary = (
            f"[{qid}] {result_b.query[:60]}... | "
            f"{query_type} · {difficulty} | "
            f"Recall: {recall_a:.2f} → {recall_b:.2f} ({sign}{delta:.2f})"
        )

        with st.expander(summary, expanded=False):
            _render_query_detail(result_a, result_b)

    if len(qids) > 20:
        st.caption(f"...and {len(qids) - 20} more")


def _render_query_detail(result_a: EvalResult, result_b: EvalResult) -> None:
    """Render drill-down detail for a single changed query."""
    # --- Query header ---
    st.markdown(f"**Query:** {result_b.query}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Type:** {result_b.query_type.value if result_b.query_type else '—'}")
    with col2:
        st.markdown(f"**Difficulty:** {result_b.difficulty.value if result_b.difficulty else '—'}")
    with col3:
        st.markdown(f"**Unanswerable:** {'Yes' if result_b.is_unanswerable else 'No'}")

    st.divider()

    # --- Retrieval diff table ---
    st.markdown("**Retrieval Diff**")
    diff_rows = compute_retrieval_diff(result_a, result_b)

    if diff_rows:
        df = pd.DataFrame(
            [
                {
                    "Chunk ID": r.chunk_id,
                    "Relevant": "Yes" if r.relevant else "No",
                    "Rank A": r.rank_a if r.rank_a is not None else "—",
                    "Rank B": r.rank_b if r.rank_b is not None else "—",
                    "Status": r.status,
                }
                for r in diff_rows
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No chunks retrieved in either run")

    # --- Answer diff (conditional) ---
    if result_a.answer and result_b.answer:
        st.divider()
        st.markdown("**Answer Diff**")

        col_a, col_b = st.columns(2)
        qid = result_a.qid

        with col_a:
            st.markdown("*Run A*")
            st.text_area(
                "Answer A",
                value=result_a.answer.text,
                height=150,
                disabled=True,
                key=f"ans_a_{qid}",
                label_visibility="collapsed",
            )
            _render_answer_metrics(result_a, prefix=f"a_{qid}")

        with col_b:
            st.markdown("*Run B*")
            st.text_area(
                "Answer B",
                value=result_b.answer.text,
                height=150,
                disabled=True,
                key=f"ans_b_{qid}",
                label_visibility="collapsed",
            )
            _render_answer_metrics(result_b, prefix=f"b_{qid}", baseline=result_a)


def _render_answer_metrics(
    result: EvalResult,
    *,
    prefix: str,
    baseline: EvalResult | None = None,
) -> None:
    """Render answer quality metrics, optionally with deltas against a baseline."""
    m = result.answer_metrics
    if not m:
        st.caption("No answer metrics")
        return

    bm = baseline.answer_metrics if baseline else None

    quality_delta = None
    correctness_delta = None
    halluc_delta = None

    if bm:
        if m.quality_score is not None and bm.quality_score is not None:
            quality_delta = f"{m.quality_score - bm.quality_score:.2f}"
        if m.correctness is not None and bm.correctness is not None:
            correctness_delta = f"{m.correctness - bm.correctness:.1f}"
        if m.hallucination_severity is not None and bm.hallucination_severity is not None:
            halluc_delta = f"{m.hallucination_severity - bm.hallucination_severity:.1f}"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Quality",
            f"{m.quality_score:.2f}" if m.quality_score is not None else "—",
            delta=quality_delta,
        )
    with c2:
        st.metric(
            "Correctness",
            f"{m.correctness:.1f}/5" if m.correctness is not None else "—",
            delta=correctness_delta,
        )
    with c3:
        st.metric(
            "Hallucination",
            f"{m.hallucination_severity:.1f}/5" if m.hallucination_severity is not None else "—",
            delta=halluc_delta,
            delta_color="inverse",  # lower hallucination is better
        )


def _compute_recall(result: EvalResult, k: int = 10) -> float:
    """Compute recall@k for a single result."""
    retrieved = set(result.retrieval_result.retrieved_chunk_ids[:k])
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
        st.write(
            f"Analyzing {trend.num_runs} runs from {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}"
        )
    else:
        st.write(f"Analyzing {trend.num_runs} runs")

    st.divider()

    # Metric selector
    col1, col2 = st.columns([1, 3])

    with col1:
        metric_choice = st.selectbox(
            "Metric",
            options=[
                "recall",
                "critical_recall",
                "weighted_recall",
                "critical_hit_rate",
                "precision",
                "ndcg",
                "mrr",
                "map",
                "quality",
                "latency",
            ],
            format_func=lambda m: {
                "recall": "Recall",
                "critical_recall": "Critical Recall",
                "weighted_recall": "Weighted Recall",
                "critical_hit_rate": "Critical Hit Rate",
                "precision": "Precision",
                "ndcg": "NDCG",
                "mrr": "MRR",
                "map": "MAP",
                "quality": "Quality Score",
                "latency": "Latency",
            }[m],
            key="trend_metric",
        )

        if metric_choice in [
            "recall",
            "critical_recall",
            "weighted_recall",
            "critical_hit_rate",
            "precision",
            "ndcg",
        ]:
            k_choice = st.selectbox(
                "K value",
                options=[1, 3, 5, 10],
                index=3,  # Default to 10
                key="trend_k",
            )
        else:
            k_choice = 10

    with col2:
        if metric_choice in [
            "recall",
            "critical_recall",
            "weighted_recall",
            "critical_hit_rate",
            "precision",
            "ndcg",
        ]:
            render_trend_chart(trend, metric=metric_choice, k=k_choice)
        else:
            render_trend_chart(trend, metric=metric_choice)

    st.divider()

    # Multi-metric comparison
    st.subheader("Multi-Metric Trend")
    render_multi_metric_trend(
        trend,
        metrics=["recall", "critical_recall", "weighted_recall", "ndcg"],
        k=10,
    )

    st.divider()

    # Summary table
    st.subheader("Run Summary Table")
    _render_trend_summary_table(trend)


def _render_trend_summary_table(trend) -> None:
    """Render summary table for trending analysis."""
    data = []
    for run in trend.runs:
        row = {
            "Timestamp": run.summary.timestamp.strftime("%Y-%m-%d %H:%M"),
            "Run": run.summary.display_name,
            "Recall@10": f"{run.aggregates.overall.recall_at_k.get(10, 0):.3f}",
            "Critical Recall@10": f"{run.aggregates.overall.critical_recall_at_k.get(10, 0):.3f}",
            "Weighted Recall@10": f"{run.aggregates.overall.weighted_recall_at_k.get(10, 0):.3f}",
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
