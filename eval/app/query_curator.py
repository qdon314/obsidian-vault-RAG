#!/usr/bin/env python3
"""
Query Curator — Streamlit App

Interactive review UI for approving, editing, and rejecting
generated evaluation queries before they enter the production
eval dataset.

Usage:
    streamlit run eval/app/query_curator.py

    Or via Makefile:
    make curate-case-queries
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from rag.eval.schema import Difficulty, QueryType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_DRAFT_PATH = PROJECT_ROOT / "eval" / "datasets" / "case_generated_queries_DRAFT.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "datasets" / "case_generated_queries.jsonl"


def load_queries(path: Path) -> list[dict]:
    """Load queries from JSONL as raw dicts (for editability)."""
    if not path.exists():
        return []
    queries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def save_queries(queries: list[dict], path: Path) -> None:
    """Save queries to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for q in queries:
            f.write(json.dumps(q, default=str) + "\n")


def main() -> None:
    st.set_page_config(
        page_title="Query Curator",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Query Curator")
    st.caption("Review, edit, and approve generated eval queries")

    # ── Sidebar: File selection & actions ────────────────────
    with st.sidebar:
        st.subheader("Input")
        draft_path = st.text_input(
            "Draft JSONL path",
            value=str(DEFAULT_DRAFT_PATH),
            key="draft_path",
        )
        draft_path = Path(draft_path)

        if st.button("Load queries", key="load_btn"):
            queries = load_queries(draft_path)
            if queries:
                st.session_state["queries"] = queries
                st.session_state["decisions"] = {
                    q["qid"]: "pending" for q in queries
                }
                st.session_state["edits"] = {}
                st.success(f"Loaded {len(queries)} queries")
            else:
                st.error(f"No queries found at {draft_path}")

        st.divider()

        # Stats
        if "queries" in st.session_state and "decisions" in st.session_state:
            decisions = st.session_state["decisions"]
            total = len(decisions)
            approved = sum(1 for v in decisions.values() if v == "approved")
            rejected = sum(1 for v in decisions.values() if v == "rejected")
            pending = sum(1 for v in decisions.values() if v == "pending")
            st.metric("Total", total)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Approved", approved)
            with col2:
                st.metric("Rejected", rejected)
            with col3:
                st.metric("Pending", pending)

            st.divider()

            # Filters
            st.subheader("Filter")
            status_filter = st.selectbox(
                "Status",
                ["all", "pending", "approved", "rejected"],
                key="status_filter",
            )
            strategy_tags = sorted(
                {
                    tag
                    for q in st.session_state["queries"]
                    for tag in q.get("tags", [])
                }
            )
            tag_filter = st.multiselect("Tags", strategy_tags, key="tag_filter")

            st.divider()

            # Batch actions
            st.subheader("Batch Actions")
            if st.button("Approve all pending"):
                for qid, status in st.session_state["decisions"].items():
                    if status == "pending":
                        st.session_state["decisions"][qid] = "approved"
                st.rerun()

            selected_tag = st.selectbox(
                "Approve all with tag:", ["", *strategy_tags], key="batch_tag"
            )
            if selected_tag and st.button(f"Approve all '{selected_tag}'"):
                for q in st.session_state["queries"]:
                    if selected_tag in q.get("tags", []):
                        st.session_state["decisions"][q["qid"]] = "approved"
                st.rerun()

            st.divider()

            # Export
            st.subheader("Export")
            output_path = st.text_input(
                "Output JSONL path",
                value=str(DEFAULT_OUTPUT_PATH),
                key="output_path",
            )
            if st.button("Export approved queries", key="export_btn"):
                approved_queries = _collect_approved()
                if approved_queries:
                    save_queries(approved_queries, Path(output_path))
                    st.success(f"Exported {len(approved_queries)} queries to {output_path}")
                else:
                    st.warning("No approved queries to export")

    # ── Main content: Query review ──────────────────────────
    if "queries" not in st.session_state:
        st.info(f"Click **Load queries** in the sidebar to start reviewing.\n\nExpected file: `{DEFAULT_DRAFT_PATH}`")
        return

    queries = st.session_state["queries"]
    decisions = st.session_state["decisions"]

    # Apply filters
    filtered = queries
    status_filter = st.session_state.get("status_filter", "all")
    if status_filter != "all":
        filtered = [q for q in filtered if decisions.get(q["qid"]) == status_filter]
    tag_filter = st.session_state.get("tag_filter", [])
    if tag_filter:
        filtered = [
            q for q in filtered if set(tag_filter) & set(q.get("tags", []))
        ]

    st.write(f"Showing {len(filtered)} of {len(queries)} queries")

    # Render each query
    for i, q in enumerate(filtered):
        _render_query_card(q, i)


def _render_query_card(q: dict, idx: int) -> None:
    """Render a single query review card."""
    qid = q["qid"]
    decisions = st.session_state["decisions"]
    current_status = decisions.get(qid, "pending")

    # Status color indicator
    status_prefix = {"approved": "+", "rejected": "-", "pending": "?"}.get(
        current_status, "?"
    )

    strategy = ", ".join(q.get("tags", []))
    difficulty = q.get("difficulty", "?")
    query_type = q.get("query_type", "?")

    label = (
        f"[{status_prefix}] {qid} | {query_type} · {difficulty} | {strategy}"
    )

    with st.expander(label, expanded=(current_status == "pending")):
        # ── Read-only metadata ──
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption(f"Source: {q.get('source_case', 'n/a')}")
        with col2:
            st.caption(f"Doc type: {q.get('case_document_type', 'n/a')}")
        with col3:
            st.caption(f"Unanswerable: {'Yes' if q.get('is_unanswerable') else 'No'}")
        with col4:
            if q.get("unanswerable_reason"):
                st.caption(f"Reason: {q['unanswerable_reason']}")

        # ── Editable query text ──
        edit_key = f"edit_query_{qid}"
        edited_query = st.text_area(
            "Query",
            value=st.session_state.get(f"edits_{qid}_query", q["query"]),
            height=80,
            key=edit_key,
        )
        if edited_query != q["query"]:
            st.session_state[f"edits_{qid}_query"] = edited_query

        # ── Editable difficulty & type ──
        col_d, col_t = st.columns(2)
        with col_d:
            diff_options = [d.value for d in Difficulty]
            current_diff = q.get("difficulty", "easy")
            st.selectbox(
                "Difficulty",
                diff_options,
                index=diff_options.index(current_diff) if current_diff in diff_options else 0,
                key=f"diff_{qid}",
            )
        with col_t:
            type_options = [t.value for t in QueryType]
            current_type = q.get("query_type", "factual")
            st.selectbox(
                "Query type",
                type_options,
                index=type_options.index(current_type) if current_type in type_options else 0,
                key=f"type_{qid}",
            )

        # ── Citations ──
        citations = q.get("relevant_citations", [])
        if isinstance(citations, set):
            citations = sorted(citations)
        st.text_input(
            "Relevant citations (comma-separated)",
            value=", ".join(citations),
            key=f"cit_{qid}",
        )

        # ── Decision buttons ──
        col_a, col_r, col_p = st.columns(3)
        with col_a:
            if st.button("Approve", key=f"approve_{qid}", type="primary"):
                decisions[qid] = "approved"
                st.rerun()
        with col_r:
            if st.button("Reject", key=f"reject_{qid}"):
                decisions[qid] = "rejected"
                st.rerun()
        with col_p:
            if st.button("Reset to pending", key=f"pending_{qid}"):
                decisions[qid] = "pending"
                st.rerun()


def _collect_approved() -> list[dict]:
    """Collect approved queries, applying any edits."""
    queries = st.session_state.get("queries", [])
    decisions = st.session_state.get("decisions", {})

    approved = []
    for q in queries:
        if decisions.get(q["qid"]) != "approved":
            continue

        # Apply edits
        out = dict(q)
        edited_query = st.session_state.get(f"edits_{q['qid']}_query")
        if edited_query is not None:
            out["query"] = edited_query

        edited_diff = st.session_state.get(f"diff_{q['qid']}")
        if edited_diff is not None:
            out["difficulty"] = edited_diff

        edited_type = st.session_state.get(f"type_{q['qid']}")
        if edited_type is not None:
            out["query_type"] = edited_type

        edited_cit = st.session_state.get(f"cit_{q['qid']}")
        if edited_cit is not None:
            out["relevant_citations"] = [
                c.strip() for c in edited_cit.split(",") if c.strip()
            ]

        approved.append(out)

    return approved


if __name__ == "__main__":
    main()
