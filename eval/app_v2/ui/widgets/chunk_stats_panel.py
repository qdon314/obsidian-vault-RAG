# eval/app_v2/ui/widgets/chunk_stats_panel.py
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.derived.chunk_stats import build_chunk_stats
from eval.app_v2.engine.domain.models import RunBundle

_MAX_CHUNKS = 20


def render_chunk_stats_panel(bundle: RunBundle) -> None:
    """Render a 'Problem chunks' expander section for the Triage page."""
    with st.expander("Problem chunks (run-wide)", expanded=False):
        if not bundle.queries:
            st.info("No queries loaded.")
            return

        stats = build_chunk_stats(bundle.queries)
        if not stats:
            st.info("No chunk data available.")
            return

        display = stats[:_MAX_CHUNKS]
        rows = [
            {
                "Chunk ID": s.chunk_id,
                "# Relevant": s.queries_where_relevant,
                "# Retrieved": s.queries_where_retrieved,
                "# Reranked": s.queries_where_reranked,
                "Miss rate": f"{s.miss_rate:.1%}",
                "Rerank drop": f"{s.rerank_drop_rate:.1%}",
            }
            for s in display
        ]
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if len(stats) > _MAX_CHUNKS:
            st.caption(f"Showing {_MAX_CHUNKS} of {len(stats)} chunks.")
