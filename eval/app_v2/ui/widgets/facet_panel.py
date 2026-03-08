# eval/app_v2/ui/widgets/facet_panel.py
"""
Stateless facet filter panel. Returns a dict[facet_key -> selected_value].
Reads FACETS and renders the correct widget per value_type automatically.
"""
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.models import AnalyzedQuery
from eval.app_v2.engine.facets.registry import FACETS, FacetDef


def _collect_values(queries: list[AnalyzedQuery], facet: FacetDef) -> list[str]:
    """Collect unique non-None values for a facet across all queries."""
    values: set[str] = set()
    for aq in queries:
        if facet.key == "severity":
            values.add(str(aq.diagnostic.severity))
        else:
            v = facet.extract(aq.record)
            if v is not None:
                values.add(str(v))
    return sorted(values)


def render_facet_panel(queries: list[AnalyzedQuery]) -> dict[str, str | bool | None]:
    """
    Render a sidebar filter panel. Returns selected values keyed by facet.key.
    Returns None for a facet if no filter is applied (show all).
    """
    st.subheader("Filters")
    selections: dict[str, str | bool | None] = {}

    for facet in FACETS:
        if facet.value_type == "enum":
            values = _collect_values(queries, facet)
            if not values:
                continue
            opts = ["(all)"] + values
            choice = st.selectbox(facet.label, opts, key=f"facet_{facet.key}")
            selections[facet.key] = None if choice == "(all)" else choice

        elif facet.value_type == "bool":
            opts = ["(all)", "True", "False"]
            choice = st.radio(facet.label, opts, horizontal=True, key=f"facet_{facet.key}")
            if choice == "True":
                selections[facet.key] = True
            elif choice == "False":
                selections[facet.key] = False
            else:
                selections[facet.key] = None

        # numeric_bucket: extend when a numeric facet is added to FACETS

    return selections
