from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.models import QueryRecord, QueryTrace


def _build_chunk_text_map(raw_data: dict) -> dict[str, str]:
    """Extract chunk_id → text from retrieved and reranked candidates.

    Iterates retrieved first, then reranked so reranked values overwrite
    (text is identical in practice, but reranked entries are post-scoring).
    """
    texts: dict[str, str] = {}
    for stage_key in ("retrieved", "reranked"):
        for cand in raw_data.get(stage_key) or []:
            if not isinstance(cand, dict):
                continue
            chunk = cand.get("chunk", {})
            if isinstance(chunk, dict):
                chunk_id = chunk.get("chunk_id")
                text = chunk.get("text", "")
                if chunk_id and text:
                    texts[chunk_id] = text
    return texts


def _render_chunk_group(
    chunk_ids: list[str],
    chunk_texts: dict[str, str],
    *,
    key_prefix: str,
) -> None:
    if not chunk_ids:
        st.caption("None.")
        return
    for cid in chunk_ids:
        text = chunk_texts.get(cid)
        header = f"`{cid}`" if text else f"`{cid}` — *no text in trace*"
        with st.expander(header, expanded=False):
            if text:
                st.text_area(
                    "",
                    value=text,
                    height=150,
                    disabled=True,
                    key=f"{key_prefix}_{cid}",
                    label_visibility="collapsed",
                )
            else:
                st.caption("Text not captured in trace.")


def render_retrieved_chunks(
    record: QueryRecord,
    trace: QueryTrace | None,
    *,
    key_prefix: str = "",
) -> None:
    """Render chunk content grouped by retrieval outcome (matched / missed / extra)
    plus the reranked stage, inside a collapsible expander.

    Pass a unique ``key_prefix`` (e.g. the query ID) to avoid Streamlit
    widget key collisions when the caller can render multiple queries.
    """
    with st.expander("Chunk content — retrieved & reranked", expanded=False):
        if trace is None:
            st.info("No trace data available for this run.")
            return

        chunk_texts = _build_chunk_text_map(trace.raw_data)

        retrieved_set = frozenset(record.retrieved_chunk_ids)
        matched = sorted(record.relevant_chunk_ids & retrieved_set)
        missed = sorted(record.relevant_chunk_ids - retrieved_set)
        extra = [cid for cid in record.retrieved_chunk_ids if cid not in record.relevant_chunk_ids]

        ns = f"{key_prefix}_" if key_prefix else ""

        st.markdown("#### Retrieved stage")
        tab_m, tab_miss, tab_e = st.tabs([
            f"Matched ✓ ({len(matched)})",
            f"Missed ✗ ({len(missed)})",
            f"Extra ({len(extra)})",
        ])

        with tab_m:
            _render_chunk_group(matched, chunk_texts, key_prefix=f"{ns}ret_matched")

        with tab_miss:
            if missed:
                st.caption(
                    "These relevant chunks were not retrieved — no text is available."
                )
            _render_chunk_group(missed, {}, key_prefix=f"{ns}ret_missed")

        with tab_e:
            _render_chunk_group(extra, chunk_texts, key_prefix=f"{ns}ret_extra")

        if trace.reranked_chunk_ids is not None:
            st.markdown("#### Reranked stage")
            _render_chunk_group(
                list(trace.reranked_chunk_ids),
                chunk_texts,
                key_prefix=f"{ns}reranked",
            )
