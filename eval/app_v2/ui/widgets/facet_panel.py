# eval/app_v2/ui/widgets/facet_panel.py
"""
Stateless facet filter panel. Returns a dict[facet_key -> selected_value].
Reads FACETS and renders the correct widget per value_type automatically.
"""

from __future__ import annotations

import contextlib

import streamlit as st

from eval.app_v2.engine.domain.models import AnalyzedQuery
from eval.app_v2.engine.facets.registry import FACETS, FacetDef


def _collect_values(queries: list[AnalyzedQuery], facet: FacetDef) -> list[str]:
    """Collect unique non-None values for an enum/bool facet across all queries."""
    values: set[str] = set()
    for aq in queries:
        if facet.key == "severity":
            values.add(str(aq.diagnostic.severity))
        elif facet.key == "diagnostic_code":
            values.add(str(aq.diagnostic.diagnostic_code))
        else:
            v = facet.extract(aq.record)
            if v is not None:
                values.add(str(v))
    return sorted(values)


def _collect_numeric_range(
    queries: list[AnalyzedQuery], facet: FacetDef
) -> tuple[float, float] | None:
    """Return (min, max) across all non-None values, or None if min >= max (no filterable range)."""
    values: list[float] = []
    for aq in queries:
        v = facet.extract(aq.record)
        if v is not None:
            with contextlib.suppress(TypeError, ValueError):
                values.append(float(v))
    if not values:
        return None
    lo, hi = min(values), max(values)
    if lo >= hi:
        return None  # no useful range to filter on
    return lo, hi


def render_facet_panel(
    queries: list[AnalyzedQuery],
) -> dict[str, str | bool | tuple[float, float] | None]:
    """
    Render a sidebar filter panel. Returns selected values keyed by facet.key.
    Returns None for a facet if no filter is applied (show all).
    For numeric_range facets, returns a tuple (lo, hi); if set to the full range it is
    treated as no filter in apply_facet_filters.
    """
    st.subheader("Filters")
    selections: dict[str, str | bool | tuple[float, float] | None] = {}

    for facet in FACETS:
        if facet.value_type == "enum":
            values = _collect_values(queries, facet)
            if not values:
                continue
            opts = ["(all)", *values]
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

        elif facet.value_type == "numeric_range":
            bounds = _collect_numeric_range(queries, facet)
            if bounds is None:
                continue  # not enough distinct values to show a slider
            lo, hi = bounds
            # Derive step from the data range: target ~100 usable steps,
            # rounded to a clean decimal, clamped to [1e-4, 1.0].
            raw_step = (hi - lo) / 100
            # Round to 4 sig-figs of the raw step to keep the slider tidy
            import math

            magnitude = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1
            step = max(1e-4, min(1.0, round(raw_step / magnitude) * magnitude))
            selected = st.slider(
                facet.label,
                min_value=float(lo),
                max_value=float(hi),
                value=(float(lo), float(hi)),
                step=step,
                key=f"facet_{facet.key}",
            )
            # If user has the slider at the full range, treat as no filter
            if selected[0] <= lo and selected[1] >= hi:
                selections[facet.key] = None
            else:
                selections[facet.key] = selected

    return selections
